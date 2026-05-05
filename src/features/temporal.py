"""
Lag and rolling-window feature engineering.

These features encode the autocorrelation structure of energy time series —
the most important signal after calendar features. At hourly resolution:

  - t-24h lag captures the same-hour-yesterday effect (strongest single lag)
  - t-48h and t-168h (1 week) capture weekly seasonality
  - t-1h and t-2h capture local momentum
  - Rolling means/stds capture trend and volatility over recent windows

All lags are computed PER TARGET COLUMN so you control exactly which
series get which features. This avoids the combinatorial explosion of
lagging every column against every other.

Features produced (example for target "load_mw")
─────────────────────────────────────────────────
  Lag features:
    load_mw_lag1     – t-1h
    load_mw_lag2     – t-2h
    load_mw_lag24    – t-24h  (same hour yesterday)
    load_mw_lag48    – t-48h
    load_mw_lag168   – t-168h (same hour last week)

  Rolling statistics (computed on the raw series before lagging):
    load_mw_roll24_mean   – 24-hour trailing mean
    load_mw_roll24_std    – 24-hour trailing std  (volatility proxy)
    load_mw_roll168_mean  – 7-day trailing mean   (weekly trend)
    load_mw_roll168_std   – 7-day trailing std

  Difference features:
    load_mw_diff1    – first difference  (t − t-1h)
    load_mw_diff24   – 24h difference   (t − t-24h), day-on-day change

Important: all lag/rolling features introduce NaN at the head of the series
(up to lag_168 = 7 days). The pipeline leaves these as NaN — they are
handled during train/test split (simply drop the first 168 rows of training).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Default lag offsets in hours
DEFAULT_LAGS = [1, 2, 24, 48, 168]

# Rolling window sizes in hours
DEFAULT_WINDOWS = [24, 168]

# Difference offsets
DEFAULT_DIFFS = [1, 24]


def add_lag_features(
    df: pd.DataFrame,
    targets: list[str],
    lags: list[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """
    Append lag features for each target column.

    Args:
        df:      DataFrame with hourly DatetimeIndex.
        targets: Column names to generate lags for.
        lags:    List of lag offsets (in rows = hours at 1h resolution).

    Returns:
        df with lag columns appended in-place.
    """
    for col in targets:
        if col not in df.columns:
            logger.warning("Lag target column '%s' not found — skipping.", col)
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag).astype("float32")
            logger.debug("  %s_lag%d", col, lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    targets: list[str],
    windows: list[int] = DEFAULT_WINDOWS,
    min_periods_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Append rolling mean and std features for each target column.

    Rolling statistics use a *closed* window ending at t-1 (i.e. they look
    backwards only) so there is no data leakage into the future.

    Args:
        df:                DataFrame with hourly DatetimeIndex.
        targets:           Column names to roll over.
        windows:           List of window sizes in hours.
        min_periods_frac:  Minimum fraction of window that must be non-NaN
                           to produce a result (avoids NaN bleed at edges).

    Returns:
        df with rolling columns appended in-place.
    """
    for col in targets:
        if col not in df.columns:
            logger.warning("Rolling target column '%s' not found — skipping.", col)
            continue
        series = df[col]
        for w in windows:
            min_p = max(1, int(w * min_periods_frac))
            # shift(1) ensures no leakage (window ends at t-1)
            rolled = series.shift(1).rolling(window=w, min_periods=min_p)
            df[f"{col}_roll{w}_mean"] = rolled.mean().astype("float32")
            df[f"{col}_roll{w}_std"]  = rolled.std().astype("float32")
            logger.debug("  %s_roll%d_mean/std", col, w)

    return df


def add_diff_features(
    df: pd.DataFrame,
    targets: list[str],
    diffs: list[int] = DEFAULT_DIFFS,
) -> pd.DataFrame:
    """
    Append first-difference features (velocity / rate-of-change).

    diff1  = series[t] − series[t-1]   → local hourly change
    diff24 = series[t] − series[t-24]  → day-on-day same-hour change

    Args:
        df:      DataFrame with hourly DatetimeIndex.
        targets: Column names to differentiate.
        diffs:   List of difference offsets in hours.

    Returns:
        df with diff columns appended in-place.
    """
    for col in targets:
        if col not in df.columns:
            logger.warning("Diff target column '%s' not found — skipping.", col)
            continue
        for d in diffs:
            df[f"{col}_diff{d}"] = df[col].diff(d).astype("float32")
            logger.debug("  %s_diff%d", col, d)

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

    Args:
        df:      Input DataFrame.
        targets: Columns to generate temporal features for.
                 Defaults to the four main OPSD series if None.
        lags:    Lag offsets (hours).
        windows: Rolling window sizes (hours).
        diffs:   Difference offsets (hours).

    Returns:
        df with all temporal features appended.
    """
    if targets is None:
        targets = [
            c for c in ["load_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw"]
            if c in df.columns
        ]

    logger.info("Adding lag features for: %s", targets)
    df = add_lag_features(df, targets, lags)

    logger.info("Adding rolling features for: %s", targets)
    df = add_rolling_features(df, targets, windows)

    logger.info("Adding diff features for: %s", targets)
    df = add_diff_features(df, targets, diffs)

    n_new = len(targets) * (len(lags) + 2 * len(windows) + len(diffs))
    logger.info("Temporal features added: ~%d new columns", n_new)

    return df
