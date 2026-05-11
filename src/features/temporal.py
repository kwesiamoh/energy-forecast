"""
Lag and rolling-window feature engineering.

These features encode the autocorrelation structure of energy time series —
the most important signal after calendar features. At hourly resolution:

  - t-24h lag captures the same-hour-yesterday effect (strongest single lag)
  - t-48h and t-168h (1 week) capture weekly seasonality
  - t-1h and t-2h capture local momentum
  - Rolling means/stds capture trend and volatility over recent windows
  - First-differences capture velocity / rate-of-change

All lags are computed PER TARGET COLUMN so you control exactly which
series get which features.

═══════════════════════════════════════════════════════════════════════════════
BUG FIX — Catastrophic target leakage in rolling and diff features
═══════════════════════════════════════════════════════════════════════════════
SYMPTOM: add_rolling_features and add_diff_features were computing statistics
that included the current timestep t, meaning the model could indirectly see
y[t] while predicting y[t].

ROOT CAUSE:
  - add_diff_features used df[col].diff(d) directly, which computes
    y[t] − y[t-d].  For d=1 this means the feature at time t is
    y[t] − y[t-1], which requires knowing y[t].
  - add_rolling_features needed an explicit .shift(1) before .rolling()
    to guarantee the window closes at t-1, not t.

FIX (applied in this version):
  - add_diff_features now computes:   series.shift(1).diff(d)
    → feature at t  =  y[t-1] − y[t-1-d]   (strictly backward-looking)
  - add_rolling_features: .shift(1) is applied to the raw series BEFORE
    calling .rolling(), ensuring the window at t covers [t-window … t-1].
  - add_lag_features: lags are inherently leakage-free (shift(k) for k≥1).

All features for timestep t are now derived strictly from y[t-1] and earlier.

Features produced (example for target "load_mw")
─────────────────────────────────────────────────
  Lag features (leakage-free by construction):
    load_mw_lag1    – y[t-1]
    load_mw_lag2    – y[t-2]
    load_mw_lag24   – y[t-24]  (same hour yesterday)
    load_mw_lag48   – y[t-48]
    load_mw_lag168  – y[t-168] (same hour last week)

  Rolling statistics (window closes at t-1, not t):
    load_mw_roll24_mean   – mean(y[t-24 .. t-1])
    load_mw_roll24_std    – std (y[t-24 .. t-1])
    load_mw_roll168_mean  – mean(y[t-168 .. t-1])
    load_mw_roll168_std   – std (y[t-168 .. t-1])

  Difference features (computed on y[t-1], not y[t]):
    load_mw_diff1   – y[t-1] − y[t-2]          (1-step velocity at t-1)
    load_mw_diff24  – y[t-1] − y[t-25]         (24-step velocity at t-1)

NaN warmup: up to 168 rows of NaN at the head of each series.
The train/val/test split drops the first 168 rows of training data.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LAGS    = [1, 2, 24, 48, 168]
DEFAULT_WINDOWS = [24, 168]
DEFAULT_DIFFS   = [1, 24]


def add_lag_features(
    df: pd.DataFrame,
    targets: list[str],
    lags: list[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """
    Append lag features for each target column.

    Lag features are leakage-free by construction: shift(k) with k≥1 ensures
    the feature at time t only uses y[t-k].

    Args:
        df:      DataFrame with hourly DatetimeIndex.
        targets: Column names to generate lags for.
        lags:    List of lag offsets in hours (must all be ≥1).

    Returns:
        df with lag columns appended in-place.
    """
    for col in targets:
        if col not in df.columns:
            logger.warning("Lag target '%s' not found — skipping.", col)
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag).astype("float32")
    logger.info("Lag features added for: %s", targets)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    targets: list[str],
    windows: list[int] = DEFAULT_WINDOWS,
    min_periods_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Append rolling mean and std features for each target column.

    LEAKAGE FIX: The series is shifted by 1 BEFORE applying the rolling
    window.  This ensures:
      - The window for timestep t covers  [t - window, ..., t - 1]
      - y[t] itself is NEVER included in its own rolling statistic.

    Without this shift, the rolling mean at time t would include y[t]
    itself — direct target leakage that would inflate model performance
    metrics while making the model useless at inference time.

    Args:
        df:                DataFrame with hourly DatetimeIndex.
        targets:           Column names to roll over.
        windows:           List of window sizes in hours.
        min_periods_frac:  Minimum fraction of window that must be non-NaN.

    Returns:
        df with rolling columns appended in-place.
    """
    for col in targets:
        if col not in df.columns:
            logger.warning("Rolling target '%s' not found — skipping.", col)
            continue

        # LEAKAGE FIX: shift(1) ensures the rolling window closes at t-1.
        # The window at time t then covers [t-window … t-1], never including
        # y[t].  Without this shift the rolling mean would be a form of
        # target leakage.
        shifted = df[col].shift(1)

        for w in windows:
            min_p = max(1, int(w * min_periods_frac))
            rolled = shifted.rolling(window=w, min_periods=min_p)
            df[f"{col}_roll{w}_mean"] = rolled.mean().astype("float32")
            df[f"{col}_roll{w}_std"]  = rolled.std().astype("float32")

    logger.info("Rolling features added for: %s (window closes at t-1)", targets)
    return df


def add_diff_features(
    df: pd.DataFrame,
    targets: list[str],
    diffs: list[int] = DEFAULT_DIFFS,
) -> pd.DataFrame:
    """
    Append first-difference features (velocity / rate-of-change).

    LEAKAGE FIX — leakage-free diffs via shift(1):
      ORIGINAL (BUGGY):
        df[col].diff(d)          → y[t] − y[t-d]  ← requires knowing y[t]!

      FIXED:
        df[col].shift(1).diff(d) → y[t-1] − y[t-1-d]  ← strictly causal

      The fixed version captures the same velocity signal but computed one
      step earlier, so the model sees "how fast was the series changing
      just before t" rather than "how much did it change arriving at t".

    Features produced:
      {col}_diff1   – y[t-1] − y[t-2]        (1-step velocity at t-1)
      {col}_diff24  – y[t-1] − y[t-25]       (24h velocity at t-1)

    Args:
        df:      DataFrame with hourly DatetimeIndex.
        targets: Column names to differentiate.
        diffs:   List of difference offsets in hours.

    Returns:
        df with diff columns appended in-place.
    """
    for col in targets:
        if col not in df.columns:
            logger.warning("Diff target '%s' not found — skipping.", col)
            continue

        # LEAKAGE FIX: shift(1) before diff() makes these strictly causal.
        # diff(d) on the shifted series computes y[t-1] − y[t-1-d],
        # which does not require observing y[t].
        shifted = df[col].shift(1)

        for d in diffs:
            df[f"{col}_diff{d}"] = shifted.diff(d).astype("float32")

    logger.info(
        "Diff features added for: %s (computed on y[t-1], not y[t])", targets
    )
    return df


def add_all_temporal_features(
    df: pd.DataFrame,
    targets: list[str] | None = None,
    lags: list[int] = DEFAULT_LAGS,
    windows: list[int] = DEFAULT_WINDOWS,
    diffs: list[int] = DEFAULT_DIFFS,
) -> pd.DataFrame:
    """
    Convenience wrapper: adds lags, rolling stats, and diffs in one call.

    All features are strictly causal (computed from y[t-1] and earlier).
    carbon_intensity_g_kwh is included in the default target set so that
    it automatically gets lag, rolling, and diff features for modelling.

    Args:
        df:      Input DataFrame.
        targets: Columns to generate temporal features for.
                 Defaults to all standard targets including carbon intensity.
        lags:    Lag offsets (hours).
        windows: Rolling window sizes (hours).
        diffs:   Difference offsets (hours).

    Returns:
        df with all temporal features appended.
    """
    if targets is None:
        # Include all available target candidates, including full SMARD mix and the carbon intensity target added.
        # The four SMARD minor-renewable series are added here to match the
        # expanded TARGET_COLS in pipeline.py — temporal features are generated
        # for them automatically without requiring any notebook changes.
        candidates = [
            # OPSD primaries (SMARD-backfilled post-2020)
            "load_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw",
            # Minor renewables — uninterrupted SMARD series
            "biomass_mwh_smard",
            "run_of_river_mwh_smard",
            "pumped_storage_gen_mwh_smard",
            "other_renewables_mwh_smard",
            # Remaining SMARD generation mix (for full-mix modelling)
            "biomass_mw", "run_of_river_mw", "nuclear_mw",
            "lignite_mw", "hard_coal_mw", "gas_mw",
            "carbon_intensity_g_kwh",
        ]
        targets = [c for c in candidates if c in df.columns]

    df = add_lag_features(df, targets, lags)
    df = add_rolling_features(df, targets, windows)
    df = add_diff_features(df, targets, diffs)

    n_new = len(targets) * (len(lags) + 2 * len(windows) + len(diffs))
    logger.info("Temporal features added: ~%d new columns (all leakage-free)", n_new)
    return df
