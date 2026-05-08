"""
Feature pipeline — Phase 2 orchestrator.

Chains all feature engineering steps in the correct order and produces
a fully-processed, model-ready DataFrame.

Pipeline steps:
  1. Load master parquet (output of Phase 1)
  2. Add calendar features          → src.features.calendar
  3. Add weather-derived features   → src.features.weather
  4. Add lag + rolling + diff feats → src.features.temporal
  5. Drop columns not used as model inputs
  6. Validate: check for inf, report NaN %
  7. Save feature parquet

═══════════════════════════════════════════════════════════════════════════════
BUG FIX — "0 Rows" Training Set (Sparse Feature Drop)
═══════════════════════════════════════════════════════════════════════════════
SYMPTOM:
  split_and_scale calls dropna(how="any") across ALL feature columns.
  Because get_feature_cols() was returning sparse columns like de_wpgt
  (wind peak gust, ~90% NaN) every row had at least one NaN → 0 rows survived.

FIX — get_feature_cols() now:
  1. Computes the NaN rate of every candidate column.
  2. Excludes any column where NaN rate > NAN_EXCLUSION_THRESHOLD (50%).
  3. Logs a WARNING listing the excluded columns so the user can audit them.

This means split_and_scale will receive a clean feature matrix where
dropna(how="any") keeps the vast majority of training rows.

═══════════════════════════════════════════════════════════════════════════════
BUG FIX — EDA Correlation Matrix NaN (disjoint time ranges)
═══════════════════════════════════════════════════════════════════════════════
SYMPTOM:
  df.corr() returns NaN for de_wpgt and de_tsun because the intersection
  of valid target data (ends Oct 2020) and valid sunshine data (starts 2022)
  is empty → Pearson correlation undefined → blank heatmap rows.

FIX — the valid_overlap_corr() helper trims the DataFrame to the time window
where BOTH the target column and the feature column have sufficient valid
data.  The notebook cells call this before .corr().

═══════════════════════════════════════════════════════════════════════════════
New target: carbon_intensity_g_kwh
═══════════════════════════════════════════════════════════════════════════════
Added to TARGET_COLS so that temporal.py automatically generates lag, rolling,
and diff features for it.  This supports the thesis's carbon-aware forecasting
component without requiring any changes to the notebook or model code.

═══════════════════════════════════════════════════════════════════════════════
Expanded TARGET_COLS — full renewable mix forecasting
═══════════════════════════════════════════════════════════════════════════════
Four additional SMARD series have been added to TARGET_COLS to support
forecasting the complete renewable generation mix:

  biomass_mwh_smard             – biomass generation (continuous SMARD series)
  run_of_river_mwh_smard        – run-of-river hydro
  pumped_storage_gen_mwh_smard  – pumped storage (generation mode only)
  other_renewables_mwh_smard    – catch-all: geothermal, waste, small hydro

The SMARD-suffixed columns are used rather than the OPSD equivalents because
they are uninterrupted from 2015 to the present day, whereas the OPSD columns
go NaN after October 2020.  Using the SMARD series ensures consistent coverage
across all three train / val / test splits without requiring the backfill logic
to be extended to these additional targets.

temporal.py's add_all_temporal_features() auto-generates lag, rolling, and
diff features for every column in TARGET_COLS, so no changes to temporal.py
are required.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import add_calendar_features
from .temporal import add_all_temporal_features
from .weather import add_weather_features

logger = logging.getLogger(__name__)

# ── Column definitions ────────────────────────────────────────────────────────

# Primary OPSD targets (backfilled from SMARD post-2020 in merge.py)
OPSD_TARGET_COLS = [
    "load_mw",
    "solar_mw",
    "wind_onshore_mw",
    "wind_offshore_mw",
]

# Full generation mix from SMARD
SMARD_TARGET_COLS = [
    # Raw SMARD generation columns (suffixed by merge.py's smard_overlay step).
    # ⚠ Keep this list in sync with SMARD_FILTERS in smard.py.
    # oil_mwh, total_*, renewable_share, carbon_intensity_g_kwh are NOT fetched
    # by smard.py (removed in the filter-ID audit) and must not appear here.
    "load_mwh_smard",
    "residual_load_mwh_smard",
    "pumped_storage_cons_mwh_smard",
    "solar_mwh_smard",
    "wind_onshore_mwh_smard",
    "wind_offshore_mwh_smard",
    "biomass_mwh_smard",
    "run_of_river_mwh_smard",
    "pumped_storage_gen_mwh_smard",
    "other_renewables_mwh_smard",
    "nuclear_mwh_smard",               # zero after April 2023
    "lignite_mwh_smard",
    "hard_coal_mwh_smard",
    "gas_mwh_smard",
    "other_conventional_mwh_smard",
    "day_ahead_price_eur_smard",
]

# Default target set — OPSD primaries (SMARD-backfilled post-2020) plus the
# four minor renewable series sourced directly from SMARD (continuous coverage
# 2015 → present, no backfill gap) plus carbon intensity for the thesis's
# carbon-aware forecasting component.
#
# SMARD columns are used for the minor renewables rather than the OPSD
# equivalents because the OPSD series go NaN after October 2020, which would
# create a large missing-data gap in the val and test splits.
TARGET_COLS: list[str] = OPSD_TARGET_COLS + [
    # Minor renewables — uninterrupted SMARD series
    "biomass_mwh_smard",
    "run_of_river_mwh_smard",
    "pumped_storage_gen_mwh_smard",
    "other_renewables_mwh_smard",
    # Derived thesis target
    "carbon_intensity_g_kwh",
]

# ── Sparse-feature exclusion threshold ───────────────────────────────────────
# Columns with more than this fraction of NaN are excluded from the feature
# matrix returned by get_feature_cols().  This prevents the dropna(how="any")
# in split_and_scale from wiping out all training rows.
NAN_EXCLUSION_THRESHOLD: float = 0.50


# ── Main pipeline function ────────────────────────────────────────────────────

def build_features(
    processed_dir: Path,
    lag_targets: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Run the full Phase 2 feature engineering pipeline.

    Args:
        processed_dir:  Directory containing master.parquet (Phase 1 output).
        lag_targets:    Columns to generate temporal features for.
                        Defaults to TARGET_COLS if None.
        force:          Recompute even if features.parquet already exists.

    Returns:
        Feature DataFrame (all original + engineered columns).
    """
    processed_dir = Path(processed_dir)
    cache = processed_dir / "features.parquet"

    if cache.exists() and not force:
        logger.info("Loading cached feature parquet from %s", cache)
        return pd.read_parquet(cache)

    master_path = processed_dir / "master.parquet"
    if not master_path.exists():
        raise FileNotFoundError(
            f"master.parquet not found at {master_path}. Run Phase 1 first."
        )
    logger.info("Loading master dataset from %s …", master_path)
    df = pd.read_parquet(master_path)
    logger.info("Master shape: %s", df.shape)

    logger.info("[2/5] Adding calendar features …")
    df = add_calendar_features(df)

    logger.info("[3/5] Adding weather-derived features …")
    df = add_weather_features(df)

    # ── Compute carbon_intensity_g_kwh ────────────────────────────────────────
    # Weighted-average grid carbon intensity using the SMARD generation mix
    # (always present and current, unlike OPSD which stops Oct 2020).
    # Formula: Σ(generation_i × emission_factor_i) / Σ(generation_i)
    # Units:   g CO₂-eq / kWh  (MWh at 1-h resolution ≡ average MW, so
    #          MWh numerator and MWh denominator cancel correctly).
    # Emission factors match _EMISSION_FACTORS_G_KWH in smard.py exactly.
    _CARBON_EF: dict[str, float] = {
        "lignite_mwh_smard":              1000.0,
        "hard_coal_mwh_smard":             820.0,
        "gas_mwh_smard":                   490.0,
        "other_conventional_mwh_smard":    700.0,   # coal/gas/oil mix proxy
        "biomass_mwh_smard":               230.0,
        "solar_mwh_smard":                  40.0,
        "wind_onshore_mwh_smard":           15.0,
        "wind_offshore_mwh_smard":          15.0,
        "run_of_river_mwh_smard":           15.0,
        "pumped_storage_gen_mwh_smard":     15.0,
        "nuclear_mwh_smard":                12.0,
        "other_renewables_mwh_smard":       40.0,
    }
    _gen_ef = {col: ef for col, ef in _CARBON_EF.items() if col in df.columns}
    if _gen_ef:
        _gen_cols = list(_gen_ef.keys())
        # no negative generation and prevent NaN propagation
        _clipped = df[_gen_cols].clip(lower=0).fillna(0)
        _total = _clipped.sum(axis=1)                  # MWh total this hour
        _co2 = sum(_clipped[col] * ef for col, ef in _gen_ef.items())
        # Avoid division by zero when all generation is NaN or zero
        df["carbon_intensity_g_kwh"] = (
            _co2.where(_total > 0) / _total.where(_total > 0)
        ).astype("float32")
        _valid = df["carbon_intensity_g_kwh"].notna().sum()
        logger.info(
            "carbon_intensity_g_kwh computed from %d SMARD columns "
            "(%d / %d rows valid).",
            len(_gen_ef), _valid, len(df),
        )
    else:
        logger.warning(
            "carbon_intensity_g_kwh: no SMARD generation columns found in "
            "master DataFrame. Column will be all-NaN. "
            "Ensure Phase 1 (build_master) ran successfully."
        )
        df["carbon_intensity_g_kwh"] = np.nan

    logger.info("[4/5] Adding temporal features …")
    targets = lag_targets or TARGET_COLS
    df = add_all_temporal_features(df, targets=targets)

    logger.info("[5/5] Validating feature DataFrame …")
    _validate(df)

    df.to_parquet(cache)
    logger.info("Feature parquet saved → %s  (shape: %s)", cache, df.shape)
    return df


def get_feature_cols(
    df: pd.DataFrame,
    nan_threshold: float = NAN_EXCLUSION_THRESHOLD,
) -> list[str]:
    """
    Return the list of columns to use as MODEL INPUTS.

    Exclusion rules (applied in order):
      1. Target columns — these are labels, not inputs.
      2. SMARD overlay columns (c.endswith("_smard")) — validation-only.
         Uses endswith() to preserve lag/rolling/diff features derived from
         SMARD targets (e.g. biomass_mwh_smard_lag24).
      3. Per-station weather columns — composites (de_*) are used instead.
      4. OPSD provenance flags (*_from_smard) — informational, not features.
      5. BUG FIX: columns with > nan_threshold fraction NaN — these cause
         dropna(how="any") in split_and_scale to eliminate all training rows.
         Key example: de_wpgt (wind peak gust) is ~90% NaN and would wipe
         the entire training set if included.
         Excluded columns are logged as WARNINGs for the user to audit.

    Args:
        df:            Feature DataFrame (output of build_features).
        nan_threshold: Columns with NaN rate above this are excluded.
                       Default: NAN_EXCLUSION_THRESHOLD (0.50).

    Returns:
        Sorted list of feature column names.
    """
    exclude: set[str] = set()

    # Rule 1: target columns (labels — never model inputs)
    exclude.update(TARGET_COLS)
    exclude.update(SMARD_TARGET_COLS)

    # Rule 2: SMARD overlay (all remaining _smard columns — validation only).
    # TARGET_COLS already contains the four SMARD minor-renewable targets, so
    # they are excluded by Rule 1 above.  Rule 2 catches every other _smard
    # overlay column that is not in TARGET_COLS / SMARD_TARGET_COLS.
    #
    # BUG FIX: use endswith("_smard") instead of "_smard" in c.
    # The old substring check accidentally matched lag/rolling/diff features
    # derived from SMARD targets (e.g. "biomass_mwh_smard_lag24"), stripping
    # the temporal history XGBoost needs to train on minor renewables.
    # endswith() only excludes the raw overlay columns themselves.
    exclude.update(c for c in df.columns if c.endswith("_smard"))

    # Rule 3: per-station weather (only use de_* composites)
    _station_slugs = [
        "berlin", "leipzig", "dresden", "bremen", "hamburg",
        "hannover", "kassel", "cologne", "dortmund", "essen",
        "frankfurt", "stuttgart", "munich", "nuremberg", "augsburg",
        "zugspitze", "flensburg", "rostock", "erfurt", "muenster",
    ]
    for slug in _station_slugs:
        exclude.update(c for c in df.columns if c.startswith(f"{slug}_"))

    # Rule 4: provenance flags
    exclude.update(c for c in df.columns if c.endswith("_from_smard"))

    # Candidate feature columns after structural exclusions
    candidates = [c for c in df.columns if c not in exclude]

    # Rule 5 (BUG FIX): drop columns that are too sparse.
    # This is the critical gate that prevents dropna(how="any") in
    # split_and_scale from producing a zero-row training set.
    # de_wpgt (wind peak gust) is the primary offender at ~90% NaN.
    nan_rates = df[candidates].isna().mean()
    too_sparse = nan_rates[nan_rates > nan_threshold].index.tolist()
    not_sparse = nan_rates[nan_rates <= nan_threshold].index.tolist()

    if too_sparse:
        logger.warning(
            "get_feature_cols: excluding %d column(s) with >%.0f%% NaN "
            "(would cause dropna to wipe training rows):\n  %s",
            len(too_sparse),
            nan_threshold * 100,
            "\n  ".join(
                f"{c}: {nan_rates[c]:.1%}" for c in sorted(too_sparse)
            ),
        )

    return sorted(not_sparse)


# ── Correlation helper (fixes EDA NaN heatmap) ───────────────────────────────

def valid_overlap_corr(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    min_valid_frac: float = 0.20,
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations on the valid overlapping timeframe.

    BUG FIX: calling df.corr() when feature columns and target columns have
    non-overlapping valid periods produces NaN (Pearson requires at least 2
    shared non-NaN observations).  This helper:
      1. Trims the DataFrame to the time range where BOTH the target column
         set AND the feature column set have ≥ min_valid_frac valid values.
      2. Drops any column (feature or target) that has > 50% NaN in the
         trimmed window before calling .corr().
      3. Returns only the [feature_cols] × [target_cols] sub-matrix.

    Args:
        df:             Feature DataFrame.
        feature_cols:   Rows of the correlation matrix.
        target_cols:    Columns of the correlation matrix.
        min_valid_frac: Minimum fraction of valid values required in the
                        overlap window for a column to be retained.

    Returns:
        DataFrame of shape (n_features, n_targets) — NaN-free where possible.
    """
    all_cols = list(set(feature_cols) | set(target_cols))
    sub = df[[c for c in all_cols if c in df.columns]].copy()

    # Find the time range where target columns have valid data
    target_valid = sub[[c for c in target_cols if c in sub.columns]].notna().any(
        axis=1)
    if not target_valid.any():
        logger.warning("valid_overlap_corr: no valid target rows found.")
        return pd.DataFrame()

    t_start = sub.index[target_valid].min()
    t_end = sub.index[target_valid].max()
    sub = sub.loc[t_start:t_end]

    # Drop columns that are still too sparse in this window
    valid_fracs = sub.notna().mean()
    keep = valid_fracs[valid_fracs >= min_valid_frac].index.tolist()
    dropped = [c for c in all_cols if c in sub.columns and c not in keep]
    if dropped:
        logger.warning(
            "valid_overlap_corr: dropping %d column(s) with <%.0f%% valid "
            "in overlap window [%s → %s]:\n  %s",
            len(dropped), min_valid_frac * 100,
            t_start.date(), t_end.date(),
            ", ".join(dropped),
        )
    sub = sub[keep]

    # Compute full correlation matrix, return only feature × target sub-matrix
    corr = sub.corr(method="pearson")
    feat_in = [c for c in feature_cols if c in corr.index]
    tgt_in = [c for c in target_cols if c in corr.columns]
    return corr.loc[feat_in, tgt_in]


# ── Validation helper ─────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> None:
    """Log warnings for inf values and high NaN rates."""
    numeric = df.select_dtypes(include=[np.floating])
    inf_cols = [c for c in numeric.columns if np.isinf(numeric[c]).any()]
    if inf_cols:
        logger.warning(
            "Inf values found in: %s — replacing with NaN.", inf_cols)
        df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)

    nan_pct = df.isna().mean().mul(100).round(1)
    high_nan = nan_pct[nan_pct > 5.0]
    if not high_nan.empty:
        logger.warning(
            "Columns with >5%% NaN (lag warmup or sparse source data):\n%s",
            high_nan.to_string(),
        )

    logger.info(
        "Validation complete. Shape: %s | Columns with any NaN: %d",
        df.shape, df.isna().any().sum(),
    )
