"""
Calendar feature engineering.

Extracts time-based signals from a UTC DatetimeIndex. These are the
cheapest but most powerful features for energy forecasting — load and
generation follow strong diurnal, weekly, and seasonal cycles.

Features produced
─────────────────
  Cyclical encodings (sin/cos pairs — preserves periodicity for neural nets):
    hour_sin, hour_cos          – hour of day  (period 24)
    dow_sin,  dow_cos           – day of week  (period 7)
    month_sin, month_cos        – month        (period 12)
    doy_sin,  doy_cos           – day of year  (period 365.25)

  Ordinal / categorical (useful for tree models and embeddings):
    hour          0–23
    dow           0 (Mon) – 6 (Sun)
    month         1–12
    quarter       1–4
    week_of_year  1–53
    day_of_year   1–366
    year          e.g. 2020

  Binary flags:
    is_weekend      1 if Saturday or Sunday
    is_night        1 if hour ∈ {22,23,0,1,2,3,4,5}
    is_peak_morning 1 if hour ∈ {7,8,9,10}   (morning demand peak DE)
    is_peak_evening 1 if hour ∈ {17,18,19,20} (evening demand peak DE)

  German public holidays (national-level):
    is_holiday      1 on public holiday dates
    is_holiday_eve  1 on the day before a public holiday

  Season:
    season   0=winter, 1=spring, 2=summer, 3=autumn  (meteorological)

All outputs are float32 (sin/cos, flags) or int16 (ordinals) to minimise
memory. The index is preserved — no rows are added or removed.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import holidays as holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    logger.warning(
        "Package 'holidays' not installed — is_holiday will be all zeros. "
        "Install with:  pip install holidays"
    )
    _HOLIDAYS_AVAILABLE = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _sin_cos(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode a cyclic variable as (sin, cos) pair."""
    angle = 2 * np.pi * values / period
    return np.sin(angle).astype("float32"), np.cos(angle).astype("float32")


def _german_holidays(years: list[int]) -> set:
    """Return a set of date objects for German public holidays."""
    if not _HOLIDAYS_AVAILABLE:
        return set()
    de_holidays = holidays_lib.Germany(years=years)
    return set(de_holidays.keys())


# ── main function ─────────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append calendar features to df in-place (returns the same object).

    Args:
        df: DataFrame with a UTC-aware DatetimeIndex.

    Returns:
        df with new calendar columns appended.
    """
    idx = df.index
    assert isinstance(idx, pd.DatetimeIndex), "Index must be a DatetimeIndex."

    # Localise to German time for correct holiday + DST alignment
    local = idx.tz_convert("Europe/Berlin")

    # ── Ordinals ──────────────────────────────────────────────────────────
    df["hour"]         = local.hour.astype("int16")
    df["dow"]          = local.dayofweek.astype("int16")          # 0=Mon
    df["month"]        = local.month.astype("int16")
    df["quarter"]      = local.quarter.astype("int16")
    df["week_of_year"] = local.isocalendar().week.astype("int16").values
    df["day_of_year"]  = local.dayofyear.astype("int16")
    df["year"]         = local.year.astype("int16")

    # ── Cyclical encodings ────────────────────────────────────────────────
    df["hour_sin"],  df["hour_cos"]  = _sin_cos(df["hour"],  24.0)
    df["dow_sin"],   df["dow_cos"]   = _sin_cos(df["dow"],   7.0)
    df["month_sin"], df["month_cos"] = _sin_cos(df["month"], 12.0)
    df["doy_sin"],   df["doy_cos"]   = _sin_cos(df["day_of_year"], 365.25)

    # ── Binary flags ──────────────────────────────────────────────────────
    df["is_weekend"]      = (df["dow"] >= 5).astype("float32")
    df["is_night"]        = df["hour"].isin(range(22, 24)).astype("float32") + \
                            df["hour"].isin(range(0, 6)).astype("float32")
    df["is_night"]        = (df["is_night"] > 0).astype("float32")
    df["is_peak_morning"] = df["hour"].isin([7, 8, 9, 10]).astype("float32")
    df["is_peak_evening"] = df["hour"].isin([17, 18, 19, 20]).astype("float32")

    # ── German public holidays ────────────────────────────────────────────
    years = list(df["year"].unique())
    holiday_dates = _german_holidays(years)
    local_dates = local.date  # numpy array of datetime.date

    df["is_holiday"] = pd.array(
        [d in holiday_dates for d in local_dates], dtype="boolean"
    ).astype("float32")

    # Eve = the day before a holiday
    next_day_dates = (local + pd.Timedelta(days=1)).date
    df["is_holiday_eve"] = pd.array(
        [d in holiday_dates for d in next_day_dates], dtype="boolean"
    ).astype("float32")

    # ── Season (meteorological: Dec-Feb=0, Mar-May=1, Jun-Aug=2, Sep-Nov=3) ─
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df["season"] = df["month"].map(season_map).astype("int16")

    logger.debug("Calendar features added: %d new columns", 22)
    return df
