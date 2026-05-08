"""
Weather-derived feature engineering.

Transforms raw Meteostat weather variables into domain-relevant energy
features.

═══════════════════════════════════════════════════════════════════════════════
BUG FIX — clearsky_index evaluates to exactly 0.00 across all timestamps
═══════════════════════════════════════════════════════════════════════════════
SYMPTOM: clearsky_index is 0.0 everywhere, failing to track the solar cycle.

ROOT CAUSE (two compounding issues):
  1. de_tsun may be ABSENT from the master DataFrame if meteostat.py dropped
     it due to excessive NaN (the DWD fallback introduced in meteostat.py v2).
     When de_tsun is None, weather.py used to silently skip the feature —
     but previously the column WAS present (all-NaN), so the assignment ran
     and produced 0.0 from NaN arithmetic with clip(0, 1).

  2. Even when de_tsun IS present, the original code wrote:
         df["clearsky_index"] = (tsun / 60.0).clip(0.0, 1.0)
     If de_tsun happened to be stored as integer minutes (0–60) and was
     inadvertently read back as int dtype after a parquet round-trip,
     Python integer division 0 // 60 == 0 for all values < 60, and even
     values of e.g. 30 (30 minutes of sun) would truncate to 0.

FIX:
  - Explicitly cast tsun to float64 BEFORE dividing by 60.0.  This guarantees
    floating-point division regardless of the dtype on disk.
  - Guard: only create clearsky_index if de_tsun is present AND has at least
    _TSUN_MIN_VALID_FRAC non-NaN values.  Otherwise skip the feature entirely.
  - Add a sanity check that the resulting clearsky_index has non-trivial
    variance (daytime hours should be > 0 wherever sun shines).

═══════════════════════════════════════════════════════════════════════════════
Promoted: pvlib Ineichen clearsky model (preferred over de_tsun proxy)
═══════════════════════════════════════════════════════════════════════════════
The Phase 1 notebook computed a more physically rigorous clearsky_index using
pvlib's Ineichen model at Germany's geographic centroid. That implementation
is now the primary path in add_weather_features():

  1. If pvlib is installed, compute clearsky_index via the Ineichen model.
     This is the correct physical approach: it gives the theoretical clear-sky
     GHI, normalised against installed solar capacity.  NaN during nighttime.
  2. If pvlib is not installed, fall back to the de_tsun/60 proxy (with the
     dtype-cast fix applied).

Moving this here rather than keeping it notebook-only ensures the feature is
always present in features.parquet and is available to Phase 3+ models without
requiring notebook re-runs or manual DataFrame patching.

Features produced
─────────────────
  Thermal comfort / demand drivers:
    hdd            – Heating Degree Hours  (max(0, 15.5 − T))
    cdd            – Cooling Degree Hours  (max(0,  T − 22.0))
    temp_abs_dev   – |T − 15.5|

  Solar irradiance proxy:
    clearsky_index – preferred: pvlib Ineichen GHI ratio at DE centroid [0, 2],
                                NaN at night.
                     fallback:  tsun [min/h] / 60 → fraction [0, 1],
                                only if pvlib unavailable and de_tsun present.

  Wind power proxy:
    wind_power_proxy – (wspd / 100)^3
    wind_category    – Beaufort ordinal 0–4

  Humidity / precipitation flags:
    is_raining  – 1 if prcp > 0.1 mm
    humid_cold  – 1 if T < 5°C AND rhum > 80%

  Rolling weather lags:
    de_temp_lag24, de_temp_lag48, de_wspd_lag24
    de_temp_roll24_mean
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HDD_BASE = 15.5
CDD_BASE = 22.0

WIND_BINS   = [0, 5, 20, 40, 75, np.inf]
WIND_LABELS = [0, 1, 2, 3, 4]

# Minimum fraction of non-NaN values required to bother creating clearsky_index
# via the de_tsun fallback path.
_TSUN_MIN_VALID_FRAC = 0.10

# Germany geographic centroid and mean altitude — used for the pvlib clearsky.
_DE_LAT, _DE_LON, _DE_ALT = 51.2, 10.4, 300.0
# Installed solar PV capacity reference (GW) — conservative to avoid
# capacity-factor inflation in early years.
_DE_SOLAR_CAPACITY_GW = 70.0

try:
    import pvlib as _pvlib
    _PVLIB_AVAILABLE = True
except ImportError:
    _pvlib = None  # type: ignore[assignment]
    _PVLIB_AVAILABLE = False
    logger.info(
        "pvlib not installed — clearsky_index will use de_tsun/60 proxy. "
        "Install with:  pip install pvlib"
    )


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append weather-derived features to df in-place.

    Expected source columns (from Meteostat composite):
        de_temp  – national mean temperature [°C]
        de_wspd  – national mean wind speed  [km/h]
        de_tsun  – national mean sunshine    [min/h]  (may be absent)
        de_prcp  – national mean precipitation [mm]
        de_rhum  – national mean relative humidity [%]

    Missing source columns are handled gracefully: the derived feature is
    silently skipped rather than raising an error.

    Returns:
        df with weather-feature columns appended.
    """
    temp = df.get("de_temp")
    wspd = df.get("de_wspd")
    prcp = df.get("de_prcp")
    rhum = df.get("de_rhum")

    # ── Thermal demand features ───────────────────────────────────────────
    if temp is not None:
        df["hdd"]          = np.maximum(0.0, HDD_BASE - temp).astype("float32")
        df["cdd"]          = np.maximum(0.0, temp - CDD_BASE).astype("float32")
        df["temp_abs_dev"] = np.abs(temp - HDD_BASE).astype("float32")

    # ── Solar proxy ─────────────────────────────────────────────────────────
    # Preferred path: pvlib Ineichen model at Germany's geographic centroid.
    # Fallback: de_tsun/60 proxy with float64 dtype-cast fix.
    _clearsky_computed = False

    if _PVLIB_AVAILABLE:
        try:
            idx = df.index
            location  = _pvlib.location.Location(
                _DE_LAT, _DE_LON, altitude=_DE_ALT, tz="UTC"
            )
            solar_pos = location.get_solarposition(idx)
            cs        = location.get_clearsky(idx, model="ineichen")
            ghi       = cs["ghi"].values.astype(np.float64)
            ghi_pos   = ghi[ghi > 0]
            if len(ghi_pos) > 0:
                ghi_p99     = np.nanpercentile(ghi_pos, 99)
                cs_fraction = ghi / ghi_p99
                solar_col = (
                    "solar_mwh_smard" if "solar_mwh_smard" in df.columns
                    else "solar_mw"
                )
                if solar_col in df.columns:
                    obs_fraction = np.clip(
                        df[solar_col].values.astype(np.float64)
                        / (_DE_SOLAR_CAPACITY_GW * 1_000.0),
                        0.0, None,
                    )
                    is_day = (solar_pos["elevation"].values > 0) & (ghi > 1.0)
                    result = np.full(len(idx), np.nan, dtype=np.float64)
                    mask   = is_day & np.isfinite(obs_fraction) & (cs_fraction > 0.01)
                    result[mask] = np.clip(
                        obs_fraction[mask] / cs_fraction[mask], 0.0, 2.0
                    )
                    ci = pd.Series(result.astype("float32"), index=idx)
                    df["clearsky_index"] = ci
                    _clearsky_computed = True
                    logger.info(
                        "clearsky_index computed via pvlib Ineichen model "
                        "(source: %s, daylight hrs: %d, mean: %.3f, std: %.3f).",
                        solar_col, int(mask.sum()),
                        float(ci.dropna().mean()), float(ci.dropna().std()),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pvlib clearsky computation failed (%s) — falling back to "
                "de_tsun proxy.", exc
            )

    if not _clearsky_computed:
        tsun_col = df.get("de_tsun")
        if tsun_col is not None:
            valid_frac = tsun_col.notna().mean()
            if valid_frac >= _TSUN_MIN_VALID_FRAC:
                tsun_float = tsun_col.astype("float64")
                clearsky   = (tsun_float / 60.0).clip(0.0, 1.0).astype("float32")
                df["clearsky_index"] = clearsky
                _clearsky_computed = True
                if clearsky.dropna().std() < 1e-6:
                    logger.warning(
                        "clearsky_index (de_tsun proxy) near-zero variance. "
                        "Check dtype: min=%.2f max=%.2f dtype=%s",
                        tsun_col.min(), tsun_col.max(), tsun_col.dtype,
                    )
                else:
                    logger.info(
                        "clearsky_index from de_tsun proxy (dtype=%s, "
                        "valid=%.1f%%, mean=%.3f). Install pvlib for Ineichen method.",
                        tsun_col.dtype, valid_frac * 100, clearsky.mean(),
                    )
            else:
                logger.warning(
                    "de_tsun is %.0f%% NaN (below %.0f%% threshold) — "
                    "clearsky_index skipped.",
                    (1 - valid_frac) * 100, _TSUN_MIN_VALID_FRAC * 100,
                )
        else:
            logger.info("de_tsun absent and pvlib unavailable — clearsky_index skipped.")

    # ── Wind power proxy ──────────────────────────────────────────────────
    if wspd is not None:
        df["wind_power_proxy"] = ((wspd / 100.0) ** 3).astype("float32")
        df["wind_category"] = pd.cut(
            wspd, bins=WIND_BINS, labels=WIND_LABELS, right=False
        ).astype("float32")

    # ── Humidity / precipitation flags ────────────────────────────────────
    if prcp is not None:
        df["is_raining"] = (prcp > 0.1).astype("float32")

    if temp is not None and rhum is not None:
        df["humid_cold"] = ((temp < 5.0) & (rhum > 80.0)).astype("float32")

    # ── Weather lags (regime persistence) ─────────────────────────────────
    for col, lags in [("de_temp", [24, 48]), ("de_wspd", [24])]:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag).astype("float32")

    # 24-hour rolling temperature mean (thermal inertia)
    if temp is not None:
        df["de_temp_roll24_mean"] = (
            temp.shift(1).rolling(24, min_periods=12).mean().astype("float32")
        )

    logger.info("Weather-derived features added.")
    return df
