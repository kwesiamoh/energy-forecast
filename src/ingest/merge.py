"""
Pipeline merge module.

Takes the three processed datasets (OPSD, SMARD, Meteostat) and produces
a single aligned master DataFrame at hourly UTC resolution, ready for
feature engineering (Phase 2).

Merge strategy:
  - OPSD is the LEFT / primary dataset (authoritative load + generation)
  - SMARD is joined as a validation overlay (suffixed _smard)
  - Meteostat national composites are joined as feature columns
  - Only the overlapping date range is kept (inner join on time axis)

Gap handling:
  - Short gaps ≤ GAP_FILL_LIMIT hours are forward-filled within each source
    before merging (already done per-source). Here we apply a final pass.
  - Remaining NaNs are flagged in a companion boolean mask DataFrame.

Output columns (example):
  load_mw, solar_mw, wind_onshore_mw, wind_offshore_mw   ← OPSD
  load_mwh_smard, solar_mwh_smard, ...                   ← SMARD overlay
  de_temp, de_wspd, de_tsun, de_prcp, ...                ← Meteostat

Saved artefacts:
  processed/master.parquet   – the merged DataFrame
  processed/master_mask.parquet – bool mask of originally-missing values
"""

import logging
from pathlib import Path

import pandas as pd

from .meteostat import load_meteostat
from .opsd import load_opsd
from .smard import load_smard

logger = logging.getLogger(__name__)

GAP_FILL_LIMIT = 3  # hours — forward-fill gaps no longer than this


def build_master(
    raw_dir: Path,
    processed_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load, align, and merge all three data sources into a master DataFrame.

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
    mask_cache = processed_dir / "master_mask.parquet"

    if master_cache.exists() and not force:
        logger.info("Loading cached master dataset from %s", master_cache)
        return pd.read_parquet(master_cache)

    # ── 1. Load each source ──────────────────────────────────────────────────
    logger.info("Loading OPSD …")
    opsd = load_opsd(raw_dir, processed_dir, force=force)

    logger.info("Loading SMARD …")
    smard = load_smard(raw_dir, processed_dir,
                       start=start, end=end, force=force)
    smard = smard.add_suffix("_smard")

    logger.info("Loading Meteostat …")
    weather = load_meteostat(processed_dir, start=start, end=end, force=force)

    # ── 2. Clip to requested date range ─────────────────────────────────────
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()

    opsd = opsd.loc[start_ts:end_ts]
    smard = smard.loc[start_ts:end_ts]
    weather = weather.loc[start_ts:end_ts]

    # ── 3. Reindex all to a clean, gapless hourly index ─────────────────────
    full_index = pd.date_range(start=start_ts, end=end_ts, freq="1h", tz="UTC")

    opsd = opsd.reindex(full_index)
    smard = smard.reindex(full_index)
    weather = weather.reindex(full_index)

    # ── 4. Track which values were originally missing ────────────────────────
    missing_mask = pd.concat(
        [opsd.isna(), smard.isna(), weather.isna()], axis=1
    )

    # ── 5. Forward-fill short gaps ───────────────────────────────────────────
    opsd = opsd.ffill(limit=GAP_FILL_LIMIT)
    smard = smard.ffill(limit=GAP_FILL_LIMIT)
    weather = weather.ffill(limit=GAP_FILL_LIMIT)

    # ── 6. Keep only Meteostat national composites to avoid column explosion ─
    weather_composite = weather[[
        c for c in weather.columns if c.startswith("de_")]]

    # ── 7. Merge ─────────────────────────────────────────────────────────────
    master = pd.concat([opsd, smard, weather_composite], axis=1)
    master.index.name = "timestamp"

    # ── 8. Summary stats ─────────────────────────────────────────────────────
    total = len(master)
    nan_pct = master.isna().mean().mul(100).round(1)
    logger.info("Master dataset: %d rows × %d columns", total, master.shape[1])
    logger.info("NaN %% per column:\n%s", nan_pct[nan_pct > 0].to_string())

    # ── 9. Persist ───────────────────────────────────────────────────────────
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
