"""
Weather-derived feature engineering.

Transforms raw Meteostat weather variables into domain-relevant energy
features. Raw temperature and wind speed are powerful on their own, but
energy demand and generation respond to non-linear and threshold effects
that these derived features capture explicitly.

Features produced
─────────────────

  Thermal comfort / demand drivers:
    hdd            – Heating Degree Hours  (max(0, 15.5 − T))
                     Proxy for space heating load. Base 15.5°C is standard
                     for German building stock (EN ISO 15927-6).
    cdd            – Cooling Degree Hours  (max(0,  T − 22.0))
                     Proxy for air-conditioning load.
    temp_abs_dev   – |T − 15.5|  (symmetric deviation from comfort band)
                     Useful as a single thermal stress feature for models
                     that don't need the HDD/CDD split.

  Solar irradiance proxy:
    clearsky_index – tsun / 60  (sunshine fraction 0–1 per hour)
                     Meteostat gives sunshine minutes per hour (tsun).
                     Dividing by 60 gives the clearsky fraction, which
                     correlates strongly with PV output after normalising
                     for solar elevation. Use as a cheap irradiance proxy
                     when GHI data is unavailable.

  Wind power proxy:
    wind_power_proxy – (wspd / 100)^3  (wind power ∝ v³, normalised)
                       Wind power output scales with the cube of wind speed.
                       This feature linearises the relationship for linear
                       models and gives non-linear models the right prior.
    wind_category   – Beaufort-inspired ordinal: 0=calm(<5), 1=light(5–20),
                       2=moderate(20–40), 3=strong(40–75), 4=storm(>75 km/h)

  Humidity / precipitation effects:
    is_raining      – 1 if prcp > 0.1 mm  (affects PV via soiling + clouds)
    humid_cold      – 1 if T < 5°C AND rhum > 80%  (icing risk for turbines)

  Rolling weather (lags capture persistence of weather regimes):
    de_temp_lag24       – temperature 24h ago
    de_wspd_lag24       – wind speed 24h ago
    de_temp_roll24_mean – 24h trailing temperature mean

All outputs are float32. NaNs are propagated from source columns
(no imputation is done here — that happens in the pipeline).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# German heating/cooling degree base temperatures (°C)
HDD_BASE = 15.5
CDD_BASE = 22.0

# Beaufort-inspired wind speed thresholds (km/h)
WIND_BINS   = [0, 5, 20, 40, 75, np.inf]
WIND_LABELS = [0, 1, 2, 3, 4]


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append weather-derived features to df in-place.

    Expects the following columns to be present (from Meteostat composite):
        de_temp   – national mean temperature [°C]
        de_wspd   – national mean wind speed  [km/h]
        de_tsun   – national mean sunshine    [min/h]
        de_prcp   – national mean precipitation [mm]
        de_rhum   – national mean relative humidity [%]

    Missing source columns are handled gracefully: the derived feature is
    silently set to NaN rather than raising an error.

    Returns:
        df with new weather-feature columns appended.
    """

    temp = df.get("de_temp")
    wspd = df.get("de_wspd")
    tsun = df.get("de_tsun")
    prcp = df.get("de_prcp")
    rhum = df.get("de_rhum")

    # ── Thermal demand features ───────────────────────────────────────────
    if temp is not None:
        df["hdd"] = np.maximum(0.0, HDD_BASE - temp).astype("float32")
        df["cdd"] = np.maximum(0.0, temp - CDD_BASE).astype("float32")
        df["temp_abs_dev"] = np.abs(temp - HDD_BASE).astype("float32")
        logger.debug("  hdd, cdd, temp_abs_dev")

    # ── Solar proxy ───────────────────────────────────────────────────────
    if tsun is not None:
        # tsun is minutes of sunshine per hour → divide by 60 → [0, 1]
        df["clearsky_index"] = (tsun / 60.0).clip(0.0, 1.0).astype("float32")
        logger.debug("  clearsky_index")

    # ── Wind power proxy ──────────────────────────────────────────────────
    if wspd is not None:
        # Normalise speed to roughly [0,1] before cubing to keep scale sane
        df["wind_power_proxy"] = ((wspd / 100.0) ** 3).astype("float32")
        df["wind_category"] = pd.cut(
            wspd, bins=WIND_BINS, labels=WIND_LABELS, right=False
        ).astype("float32")
        logger.debug("  wind_power_proxy, wind_category")

    # ── Humidity / precipitation flags ────────────────────────────────────
    if prcp is not None:
        df["is_raining"] = (prcp > 0.1).astype("float32")
        logger.debug("  is_raining")

    if temp is not None and rhum is not None:
        df["humid_cold"] = ((temp < 5.0) & (rhum > 80.0)).astype("float32")
        logger.debug("  humid_cold")

    # ── Weather lags (regime persistence) ─────────────────────────────────
    for col, lags in [("de_temp", [24, 48]), ("de_wspd", [24])]:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag).astype("float32")
                logger.debug("  %s_lag%d", col, lag)

    # 24-hour rolling temperature mean (thermal inertia of buildings)
    if temp is not None:
        df["de_temp_roll24_mean"] = (
            temp.shift(1).rolling(24, min_periods=12).mean().astype("float32")
        )
        logger.debug("  de_temp_roll24_mean")

    logger.info("Weather-derived features added.")
    return df
