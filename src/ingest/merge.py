"""
Pipeline merge module — Phase 1 output.

Takes the three processed datasets (OPSD, SMARD, Meteostat) and produces
a single aligned master DataFrame at hourly UTC resolution, ready for
feature engineering (Phase 2).

═══════════════════════════════════════════════════════════════════════════════
Architecture — separation of concerns
═══════════════════════════════════════════════════════════════════════════════
opsd.py   → pure OPSD download + parse (no backfill logic)
smard.py  → pure SMARD download + parse
merge.py  → owns ALL cross-source logic, including the OPSD ← SMARD backfill

The SMARD backfill was previously split between opsd.py and merge.py.  It now
lives entirely here so that opsd.py remains a clean single-responsibility
loader.

═══════════════════════════════════════════════════════════════════════════════
OPSD stale data (target columns 100% NaN from Oct 2020 onward)
═══════════════════════════════════════════════════════════════════════════════
OPSD stopped being updated in October 2020.  After that date load_mw,
solar_mw, wind_onshore_mw and wind_offshore_mw from OPSD are all NaN.

Fix: after the standard left-join, a backfill pass maps the corresponding
SMARD columns (which are current through today) into the OPSD target columns
for any hour where OPSD is missing.

    OPSD target ← OPSD value   if not NaN (authoritative through ~Oct 2020)
    OPSD target ← SMARD value  if OPSD is NaN (current data from Oct 2020 →)

Unit note: SMARD reports MWh per hour.  At 1-hour resolution
MWh ≡ average MW for that hour, so the numeric values are directly
comparable.  The filled columns stay named *_mw to signal mean-power
semantics, and a boolean companion column *_from_smard marks the filled rows.

═══════════════════════════════════════════════════════════════════════════════
Merge strategy
═══════════════════════════════════════════════════════════════════════════════
Layer 1 (primary actuals)  : OPSD   → load_mw, solar_mw, wind_*_mw, ...
                              (+ SMARD backfill post-2020, applied here)
Layer 2 (validation / fill): SMARD  → all columns suffixed _smard
Layer 3 (weather)          : Meteostat → de_temp, de_wspd, de_tsun, ...
                              (de_tsun may be absent — see meteostat.py)

Gap handling:
  - Short gaps ≤ GAP_FILL_LIMIT hours are forward-filled per-source.
  - A companion boolean mask DataFrame records which values were originally
    missing before gap-filling or SMARD backfill.

Saved artefacts:
  processed/master.parquet       – merged DataFrame
  processed/master_mask.parquet  – bool mask of originally-missing values
"""

import logging
from pathlib import Path

import pandas as pd

from .meteostat import load_meteostat
from .opsd import load_opsd
from .smard import load_smard

logger = logging.getLogger(__name__)

GAP_FILL_LIMIT = 3  # hours — forward-fill gaps no longer than this

# OPSD target column → corresponding SMARD column for backfill.
# This mapping is owned by merge.py (not opsd.py) because it encodes
# cross-source knowledge that belongs to the orchestration layer.
_OPSD_SMARD_BACKFILL_MAP: dict[str, str] = {
    "load_mw":          "load_mwh",
    "solar_mw":         "solar_mwh",
    "wind_onshore_mw":  "wind_onshore_mwh",
    "wind_offshore_mw": "wind_offshore_mwh",
    # Extend here if OPSD gains extra columns in a future release:
    # "biomass_mw": "biomass_mwh",
}


def _backfill_opsd_from_smard(
    opsd: pd.DataFrame,
    smard: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing OPSD target columns with the corresponding SMARD series.

    This function is called by build_master() after both sources have been
    loaded and reindexed.  It is NOT called from opsd.py — the backfill is
    a merge-time concern, not an ingestion-time concern.

    For each (opsd_col, smard_col) pair in _OPSD_SMARD_BACKFILL_MAP:
      - Where opsd_col is NaN, copy the value from smard_col.
      - Add a boolean column {opsd_col}_from_smard that is True for filled rows.

    Both DataFrames must share the same DatetimeIndex.

    Returns:
        A copy of opsd with gaps filled and provenance flags added.
    """
    opsd = opsd.copy()
    for opsd_col, smard_col in _OPSD_SMARD_BACKFILL_MAP.items():
        if opsd_col not in opsd.columns:
            logger.warning(
                "OPSD backfill: column '%s' not in OPSD — skipping.", opsd_col
            )
            continue
        if smard_col not in smard.columns:
            logger.warning(
                "OPSD backfill: SMARD column '%s' not available — skipping.", smard_col
            )
            continue

        missing_mask = opsd[opsd_col].isna()
        n_missing = missing_mask.sum()

        if n_missing == 0:
            opsd[f"{opsd_col}_from_smard"] = False
            continue

        # Align SMARD values to OPSD index (inner, so extra SMARD rows are ignored)
        smard_aligned = smard[smard_col].reindex(opsd.index)
        opsd.loc[missing_mask, opsd_col] = smard_aligned[missing_mask].values
        opsd[f"{opsd_col}_from_smard"] = missing_mask.astype(bool)

        n_filled = (~opsd[opsd_col].isna() & missing_mask).sum()
        logger.info(
            "OPSD backfill: '%s' ← '%s': filled %d / %d missing rows.",
            opsd_col, smard_col, n_filled, n_missing,
        )

    return opsd


def build_master(
    raw_dir: Path,
    processed_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load, align, and merge all three data sources into a master DataFrame.

    The SMARD backfill for OPSD's post-2020 gap is applied here (not in
    opsd.py), keeping each ingestion module focused on its own source.

    Args:
        raw_dir:        Directory where raw downloads are stored.
        processed_dir:  Directory for processed parquet files (and master).
        start:          ISO date — clip data before this date.
        end:            ISO date — clip data after this date (None = today).
        force:          Re-run the full pipeline even if cached.

    Returns:
        Hourly master DataFrame indexed by UTC DatetimeIndex.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    master_cache = processed_dir / "master.parquet"
    mask_cache   = processed_dir / "master_mask.parquet"

    if master_cache.exists() and not force:
        logger.info("Loading cached master dataset from %s", master_cache)
        return pd.read_parquet(master_cache)

    # ── 1. Load each source ──────────────────────────────────────────────
    logger.info("Loading OPSD …")
    opsd = load_opsd(raw_dir, processed_dir, force=force)

    logger.info("Loading SMARD …")
    smard = load_smard(raw_dir, processed_dir, start=start, end=end, force=force)

    logger.info("Loading Meteostat …")
    weather = load_meteostat(processed_dir, start=start, end=end, force=force)

    # ── 2. Clip to requested date range ──────────────────────────────────
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()

    opsd    = opsd.loc[start_ts:end_ts]
    smard   = smard.loc[start_ts:end_ts]
    weather = weather.loc[start_ts:end_ts]

    # ── 3. Reindex all to a clean, gapless hourly index ─────────────────
    full_index = pd.date_range(start=start_ts, end=end_ts, freq="1h", tz="UTC")

    opsd    = opsd.reindex(full_index)
    smard   = smard.reindex(full_index)
    weather = weather.reindex(full_index)

    # ── 4. Track originally-missing values (before any filling) ─────────
    missing_mask = pd.concat(
        [opsd.isna(), smard.isna(), weather.isna()], axis=1
    )

    # ── 5. Forward-fill short gaps within each source ────────────────────
    opsd    = opsd.ffill(limit=GAP_FILL_LIMIT)
    smard   = smard.ffill(limit=GAP_FILL_LIMIT)
    weather = weather.ffill(limit=GAP_FILL_LIMIT)

    # ── 6. OPSD ← SMARD backfill for post-Oct-2020 gap ──────────────────
    #
    #  OPSD stopped publishing after ~October 2020.  load_mw / solar_mw /
    #  wind_onshore_mw / wind_offshore_mw go to 100% NaN from that point.
    #  We fill them from the equivalent SMARD columns, which are current.
    #
    #  This logic lives here (merge.py) rather than in opsd.py so that each
    #  ingestion module stays focused on its own data source.
    #
    #  Unit note: SMARD is MWh/h which equals average MW at hourly resolution,
    #  so values are directly comparable despite the column name difference.
    logger.info("Applying OPSD ← SMARD backfill for missing target columns …")
    opsd = _backfill_opsd_from_smard(opsd, smard)

    # ── 7. Build SMARD overlay (validation columns, suffixed _smard) ─────
    smard_overlay = smard.add_suffix("_smard")

    # ── 8. Keep only Meteostat national composites to avoid explosion ────
    weather_composite = weather[[
        c for c in weather.columns if c.startswith("de_")
    ]]

    # ── 9. Merge all layers ───────────────────────────────────────────────
    master = pd.concat([opsd, smard_overlay, weather_composite], axis=1)
    master.index.name = "timestamp"

    # ── 10. Summary stats ─────────────────────────────────────────────────
    total   = len(master)
    nan_pct = master.isna().mean().mul(100).round(1)
    logger.info("Master dataset: %d rows × %d columns", total, master.shape[1])
    high_nan = nan_pct[nan_pct > 0]
    if not high_nan.empty:
        logger.info("NaN %% per column (non-zero only):\n%s", high_nan.to_string())

    # ── 11. Persist ───────────────────────────────────────────────────────
    master.to_parquet(master_cache)
    missing_mask.to_parquet(mask_cache)
    logger.info("Saved master → %s", master_cache)
    logger.info("Saved missing mask → %s", mask_cache)

    return master


def load_master(processed_dir: Path) -> pd.DataFrame:
    """Convenience loader — just reads the cached master parquet."""
    path = processed_dir / "master.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {path}. Run build_master() first."
        )
    return pd.read_parquet(path)
