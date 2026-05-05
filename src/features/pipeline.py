"""
Feature pipeline — Phase 2 orchestrator.

Chains all feature engineering steps in the correct order and produces
a fully-processed, model-ready DataFrame.

Pipeline steps (in order):
  1. Load master parquet (output of Phase 1)
  2. Add calendar features          → src.features.calendar
  3. Add weather-derived features   → src.features.weather
  4. Add lag + rolling + diff feats → src.features.temporal
  5. Drop columns not used as model inputs
  6. Validate: check for inf, report NaN%
  7. Save feature parquet to processed/features.parquet

The pipeline is idempotent: running it twice with force=False reads the
cache. Set force=True to recompute (e.g. after changing feature definitions).

Column groups (for downstream use in models):
  TARGET_COLS  – what we want to predict
  FEATURE_COLS – model inputs (everything else except SMARD overlay cols)
  DROP_COLS    – columns excluded from the feature set
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

TARGET_COLS = [
    "load_mw",
    "solar_mw",
    "wind_onshore_mw",
    "wind_offshore_mw",
]

# SMARD overlay is for validation — not a model input
SMARD_COLS_PREFIX = "smard"

# Columns to exclude from the feature matrix (but kept in the saved parquet
# so you can still access them for post-hoc analysis)
ANALYSIS_ONLY_COLS = [c for c in [] if True]  # extend as needed


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
                        Feature parquet is also saved here.
        lag_targets:    Which columns to generate temporal features for.
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

    # ── Step 1: Load master ───────────────────────────────────────────────
    master_path = processed_dir / "master.parquet"
    if not master_path.exists():
        raise FileNotFoundError(
            f"master.parquet not found at {master_path}. Run Phase 1 first."
        )
    logger.info("Loading master dataset from %s …", master_path)
    df = pd.read_parquet(master_path)
    logger.info("Master shape: %s", df.shape)

    # ── Step 2: Calendar features ─────────────────────────────────────────
    logger.info("[2/5] Adding calendar features …")
    df = add_calendar_features(df)

    # ── Step 3: Weather-derived features ──────────────────────────────────
    logger.info("[3/5] Adding weather-derived features …")
    df = add_weather_features(df)

    # ── Step 4: Lag + rolling + diff features ─────────────────────────────
    logger.info("[4/5] Adding temporal features …")
    targets = lag_targets or TARGET_COLS
    df = add_all_temporal_features(df, targets=targets)

    # ── Step 5: Sanity checks ─────────────────────────────────────────────
    logger.info("[5/5] Validating feature DataFrame …")
    _validate(df)

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_parquet(cache)
    logger.info("Feature parquet saved → %s  (shape: %s)", cache, df.shape)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the list of columns that should be used as MODEL INPUTS.

    Excludes:
      - Target columns (these are labels, not inputs)
      - SMARD overlay columns (validation-only)
      - Raw Meteostat per-station columns (only composites are used)
    """
    exclude = set(TARGET_COLS)
    exclude.update(c for c in df.columns if "_smard" in c)
    # Per-station weather cols (berlin_temp, etc.) — composites start with de_
    station_names = ["berlin", "frankfurt", "munich", "hamburg", "stuttgart"]
    for station in station_names:
        exclude.update(c for c in df.columns if c.startswith(f"{station}_"))

    return [c for c in df.columns if c not in exclude]


# ── Validation helper ─────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> None:
    """Log warnings for inf values and high NaN rates."""
    # Inf check
    numeric = df.select_dtypes(include=[np.floating])
    inf_cols = [c for c in numeric.columns if np.isinf(numeric[c]).any()]
    if inf_cols:
        logger.warning("Inf values found in columns: %s — replacing with NaN.", inf_cols)
        df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)

    # NaN report
    nan_pct = df.isna().mean().mul(100).round(1)
    high_nan = nan_pct[nan_pct > 5.0]
    if not high_nan.empty:
        logger.warning(
            "Columns with >5%% NaN (likely due to lag warmup or missing data):\n%s",
            high_nan.to_string(),
        )

    logger.info(
        "Validation complete. Shape: %s | Columns with any NaN: %d",
        df.shape,
        df.isna().any().sum(),
    )
