"""
Meteostat weather ingestion module.

Meteostat provides historical weather and climate data from meteorological
stations worldwide via a free Python library backed by a public CDN.

Library: https://github.com/meteostat/meteostat-python
License: CC BY-NC 4.0 (attribution + non-commercial)
Attribution: Weather data by Meteostat (https://meteostat.net) under CC BY-NC 4.0

Used in this project as FEATURE ENRICHMENT for energy forecasts.
Weather is the strongest exogenous driver of:
  - Solar PV output  → surface solar radiation / cloud cover
  - Wind generation  → wind speed at hub height
  - Load             → temperature (heating/cooling degree days)

Station strategy:
  We use a small set of representative German stations covering the main
  climate zones. Hourly data is fetched per station, then averaged to a
  national composite. For more granular modelling, swap in a spatial
  interpolation layer later.

  Selected stations (DWD / WMO IDs via Meteostat):
    10384 – Berlin-Tempelhof     (northeast)
    10501 – Frankfurt/Main       (centre-west)
    10637 – München-Flughafen    (south/Bavaria)
    10400 – Hamburg-Fuhlsbüttel  (north)
    10605 – Stuttgart            (southwest)

Variables fetched (hourly):
    temp   – Air temperature [°C]
    dwpt   – Dew point [°C]
    rhum   – Relative humidity [%]
    prcp   – Precipitation [mm]
    snow   – Snow depth [mm]
    wdir   – Wind direction [°]
    wspd   – Wind speed [km/h]
    wpgt   – Wind peak gust [km/h]
    pres   – Sea-level air pressure [hPa]
    tsun   – Sunshine duration [min/h]
    coco   – Weather condition code
"""

import logging
from pathlib import Path

import pandas as pd
from meteostat import Hourly, Point, Stations

logger = logging.getLogger(__name__)

# Representative German stations: (name, lat, lon, alt_m)
DE_STATIONS = [
    ("berlin",    52.47, 13.40,  50),
    ("frankfurt", 50.03,  8.55, 111),
    ("munich",    48.35, 11.79, 447),
    ("hamburg",   53.63,  9.99,  15),
    ("stuttgart", 48.69,  9.22, 371),
]

# Columns to keep from Meteostat hourly output
KEEP_COLS = ["temp", "dwpt", "rhum", "prcp", "wspd", "wpgt", "pres", "tsun"]


def _fetch_station(
    name: str,
    lat: float,
    lon: float,
    alt: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch hourly data from the nearest Meteostat station to (lat, lon).

    Falls back gracefully if the station has gaps — Meteostat performs
    model-based interpolation internally when `model=True` (default).
    """
    location = Point(lat, lon, alt)
    data = Hourly(location, start=start, end=end)
    data = data.fetch()

    if data.empty:
        logger.warning("No Meteostat data for station %s", name)
        return pd.DataFrame()

    # Keep only desired columns that are actually present
    available = [c for c in KEEP_COLS if c in data.columns]
    df = data[available].copy()
    df.index = df.index.tz_localize("UTC")
    df.columns = [f"{name}_{c}" for c in df.columns]
    return df


def load_meteostat(
    processed_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly weather for all representative German stations and return
    a DataFrame with:
      1. Per-station columns (e.g. berlin_temp, frankfurt_wspd …)
      2. National composite columns (mean across stations, prefix 'de_')

    The composite is the main input to the forecasting model. Per-station
    columns are kept for spatial analysis and ablation studies.

    Args:
        processed_dir:  Where to cache the parquet file.
        start:          ISO date — fetch from here.
        end:            ISO date — fetch up to here (None = today).
        force:          Ignore cache.

    Returns:
        Hourly DataFrame indexed by UTC DatetimeIndex.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / "meteostat.parquet"

    if cache.exists() and not force:
        logger.info("Loading cached Meteostat parquet from %s", cache)
        return pd.read_parquet(cache)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now()

    logger.info(
        "Fetching Meteostat weather for %d stations (%s → %s) …",
        len(DE_STATIONS),
        start_ts.date(),
        end_ts.date(),
    )

    station_dfs = []
    for name, lat, lon, alt in DE_STATIONS:
        logger.info("  station: %s (%.2f, %.2f)", name, lat, lon)
        df = _fetch_station(name, lat, lon, alt, start_ts, end_ts)
        if not df.empty:
            station_dfs.append(df)

    if not station_dfs:
        raise RuntimeError("No weather data retrieved from any station.")

    merged = pd.concat(station_dfs, axis=1)
    merged = merged.sort_index()

    # Build national composite by averaging numeric variables across stations
    for var in KEEP_COLS:
        station_cols = [c for c in merged.columns if c.endswith(f"_{var}")]
        if station_cols:
            merged[f"de_{var}"] = merged[station_cols].mean(
                axis=1).astype("float32")

    # Forward-fill short gaps (≤ 2 h) — common in station data
    merged = merged.ffill(limit=2)

    logger.info(
        "Meteostat loaded: %d rows, %s → %s",
        len(merged),
        merged.index.min().date(),
        merged.index.max().date(),
    )

    merged.to_parquet(cache)
    logger.info("Cached Meteostat parquet → %s", cache)
    return merged
