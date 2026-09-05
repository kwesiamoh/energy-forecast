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
Sparse feature handling
═══════════════════════════════════════════════════════════════════════════════
get_feature_cols():
  1. Computes the NaN rate of every candidate column in the training period.
  2. Excludes any column where NaN rate > NAN_EXCLUSION_THRESHOLD (50%).
  3. Logs a WARNING listing the excluded columns so the user can audit them.

This prevents sparse sensors such as de_wpgt from entering model feature sets
without adequate training-period coverage.

═══════════════════════════════════════════════════════════════════════════════
EDA correlation with disjoint time ranges
═══════════════════════════════════════════════════════════════════════════════
The valid_overlap_corr() helper trims the DataFrame to the time window
where BOTH the target column and the feature column have sufficient valid
data.  The notebook cells call this before .corr().

═══════════════════════════════════════════════════════════════════════════════
Carbon-intensity target
═══════════════════════════════════════════════════════════════════════════════
TARGET_COLS includes carbon_intensity_g_kwh, so temporal.py generates its lag,
rolling, and difference features.

═══════════════════════════════════════════════════════════════════════════════
Renewable-mix targets
═══════════════════════════════════════════════════════════════════════════════
Four SMARD series in TARGET_COLS support renewable generation-mix forecasting:

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

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import add_calendar_features
from .temporal import add_all_temporal_features
from .weather import add_weather_features

logger = logging.getLogger(__name__)

FEATURE_CACHE_VERSION = 2
DEFAULT_TRAIN_END = "2021-12-31"

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
    # by smard.py and must not appear here.
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
# 2015 → present, no backfill gap) plus carbon intensity for carbon-aware
# forecasting experiments.
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
    # Derived carbon-aware forecasting target
    "carbon_intensity_g_kwh",
]

# ── Sparse-feature exclusion threshold ───────────────────────────────────────
# Columns with more than this fraction of NaN are excluded from the feature
# matrix returned by get_feature_cols().
NAN_EXCLUSION_THRESHOLD: float = 0.50

# Regional OPSD series are retained for diagnostics but excluded from model
# features because they do not provide consistent forecast-period coverage.
REGIONAL_OPSD_PREFIXES: tuple[str, ...] = (
    "50hertz_",
    "amprion_",
    "lu_",
    "tennet_",
    "transnetbw_",
)


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
    cache_meta = processed_dir / "features.meta.json"
    master_path = processed_dir / "master.parquet"
    requested_targets = TARGET_COLS if lag_targets is None else lag_targets
    cache_spec = {
        "version": FEATURE_CACHE_VERSION,
        "lag_targets": requested_targets,
        "master_mtime_ns": master_path.stat().st_mtime_ns if master_path.exists() else None,
    }

    cache_valid = False
    if cache.exists() and cache_meta.exists() and not force:
        try:
            cache_valid = json.loads(cache_meta.read_text()) == cache_spec
        except (OSError, ValueError):
            cache_valid = False

    if cache_valid:
        logger.info("Loading cached feature parquet from %s", cache)
        return pd.read_parquet(cache)

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

    # Carbon intensity is computed once by the SMARD loader. The merge layer
    # suffixes SMARD columns, so expose that authoritative series as a target.
    carbon_source = "carbon_intensity_g_kwh_smard"
    if carbon_source in df.columns:
        df["carbon_intensity_g_kwh"] = df[carbon_source].astype("float32")
    else:
        logger.warning(
            "carbon_intensity_g_kwh: SMARD-derived source column not found. "
            "Column will be all-NaN. "
            "Ensure Phase 1 (build_master) ran successfully."
        )
        df["carbon_intensity_g_kwh"] = np.nan

    logger.info("[4/5] Adding temporal features …")
    targets = requested_targets
    df = add_all_temporal_features(df, targets=targets)

    logger.info("[5/5] Validating feature DataFrame …")
    _validate(df)

    df.to_parquet(cache)
    cache_meta.write_text(json.dumps(cache_spec, indent=2))
    logger.info("Feature parquet saved → %s  (shape: %s)", cache, df.shape)
    return df


def get_feature_cols(
    df: pd.DataFrame,
    nan_threshold: float = NAN_EXCLUSION_THRESHOLD,
    train_end: str = DEFAULT_TRAIN_END,
) -> list[str]:
    """
    Return the list of columns to use as MODEL INPUTS.

    Exclusion rules (applied in order):
      1. Target columns — these are labels, not inputs.
      2. SMARD overlay columns (c.endswith("_smard")) — validation-only.
         Uses endswith() to preserve lag/rolling/diff features derived from
         SMARD targets (e.g. biomass_mwh_smard_lag24).
      3. Per-station weather columns — composites (de_*) are used instead.
      4. Regional OPSD columns — retained for diagnostics, not model inputs.
      5. OPSD provenance flags (*_from_smard) — informational, not features.
      6. Columns with > nan_threshold fraction NaN — insufficient training
         coverage for use as model inputs. Excluded columns are logged as
         WARNINGs for the user to audit.

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
    # endswith() excludes raw overlay columns while retaining temporal
    # derivatives such as "biomass_mwh_smard_lag24".
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

    # Rule 4: regional OPSD series retained only for diagnostics and EDA
    exclude.update(
        c for c in df.columns
        if c.startswith(REGIONAL_OPSD_PREFIXES)
    )

    # Rule 5: provenance flags
    exclude.update(c for c in df.columns if c.endswith("_from_smard"))

    # Realized same-hour aggregate wind generation is retained for diagnostics
    # and electricity-system characterization, but is not forecast-safe input.
    exclude.add("wind_mw")

    # This diagnostic uses observed solar generation and is retained for EDA,
    # but it is not available as an operational forecasting covariate.
    exclude.add("clearsky_index")

    # Candidate feature columns after structural exclusions
    candidates = [c for c in df.columns if c not in exclude]

    # Rule 6: drop columns that are too sparse for model use.
    # de_wpgt (wind peak gust) is the primary offender at ~90% NaN.
    train_end_ts = pd.Timestamp(train_end)
    if len(train_end.strip()) == 10:
        train_end_ts += pd.Timedelta(days=1)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        if train_end_ts.tzinfo is None:
            train_end_ts = train_end_ts.tz_localize(df.index.tz)
        else:
            train_end_ts = train_end_ts.tz_convert(df.index.tz)
    selection_df = df.loc[df.index < train_end_ts]
    if selection_df.empty:
        raise ValueError("No rows fall inside the feature-selection training period.")
    nan_rates = selection_df[candidates].isna().mean()
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

    Calling df.corr() when feature columns and target columns have
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
