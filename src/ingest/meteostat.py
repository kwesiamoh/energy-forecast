"""
Meteostat weather ingestion module — compatible with meteostat v2.x.

The meteostat library had a breaking API change in v2.0:
  OLD (v1.x):  from meteostat import Hourly, Point  → Hourly(point, start, end).fetch()
  NEW (v2.x):  import meteostat as ms               → ms.hourly(station, start, end).fetch()

This module uses the v2.x API. Install with:
    pip install meteostat

Data source: https://dev.meteostat.net
License: CC BY-NC 4.0
Attribution: Weather data by Meteostat (https://meteostat.net) under CC BY-NC 4.0

Direct bulk download (no API key needed, very reliable):
    https://data.meteostat.net/hourly/{year}/{station_id}.csv.gz

DWD station IDs used (5-digit WMO codes):
    10384 – Berlin Tempelhof
    10501 – Frankfurt/Main
    10637 – München-Flughafen
    10400 – Hamburg-Fuhlsbüttel
    10605 – Stuttgart (Echterdingen)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# (name, WMO station ID, lat, lon, alt_m)
DE_STATIONS = [
    ("berlin",    "10384", 52.47, 13.40,  50),
    ("frankfurt", "10501", 50.03,  8.55, 111),
    ("munich",    "10637", 48.35, 11.79, 447),
    ("hamburg",   "10400", 53.63,  9.99,  15),
    ("stuttgart", "10605", 48.69,  9.22, 371),
]

KEEP_COLS = ["temp", "dwpt", "rhum", "prcp", "wspd", "wpgt", "pres", "tsun"]


def _fetch_station_v2(
    station_id: str,
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch hourly data using meteostat v2.x API.

    Falls back to direct bulk CSV download if the library call fails.
    """
    try:
        import meteostat as ms
    except ImportError:
        raise ImportError("Install meteostat: pip install meteostat")

    try:
        # v2.x API: ms.hourly(station_id, start, end)
        ts  = ms.hourly(station_id, start, end)
        df  = ts.fetch()
    except Exception as e:
        logger.warning("meteostat v2 call failed for %s (%s): %s — trying bulk download", name, station_id, e)
        df = _fetch_bulk_csv(station_id, name, start, end)

    if df is None or df.empty:
        logger.warning("No data for station %s (%s)", name, station_id)
        return pd.DataFrame()

    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # v2 index is tz-naive (local time) → convert to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    df.columns = [f"{name}_{c}" for c in df.columns]
    return df.astype("float32")


def _fetch_bulk_csv(
    station_id: str,
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """
    Fallback: download annual bulk CSV files directly from Meteostat CDN.

    URL pattern: https://data.meteostat.net/hourly/{year}/{station_id}.csv.gz
    No API key needed.
    """
    import io
    import requests

    years = range(start.year, end.year + 1)
    col_names = ["date", "hour", "temp", "dwpt", "rhum", "prcp",
                 "snow", "wdir", "wspd", "wpgt", "pres", "tsun", "coco"]
    frames = []

    for year in years:
        url = f"https://data.meteostat.net/hourly/{year}/{station_id}.csv.gz"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 404:
                logger.debug("  bulk %s/%d: 404 (no data)", station_id, year)
                continue
            resp.raise_for_status()
            df_year = pd.read_csv(
                io.BytesIO(resp.content),
                compression="gzip",
                header=None,
                names=col_names,
                parse_dates={"timestamp": ["date", "hour"]},
            )
            df_year = df_year.set_index("timestamp")
            frames.append(df_year)
            logger.debug("  bulk %s/%d: %d rows", station_id, year, len(df_year))
        except Exception as e:
            logger.warning("  bulk %s/%d failed: %s", station_id, year, e)

    if not frames:
        return None

    df = pd.concat(frames).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)

    # Clip to requested range
    df = df.loc[start:end]
    return df[[c for c in KEEP_COLS if c in df.columns]].astype("float32")


def load_meteostat(
    processed_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly weather for all representative German stations and return
    a DataFrame with per-station columns + national composite (de_*).

    Args:
        processed_dir: Where to cache the parquet file.
        start:         ISO date string.
        end:           ISO date string (None = today).
        force:         Ignore cache and re-download.

    Returns:
        Hourly DataFrame indexed by UTC DatetimeIndex.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / "meteostat.parquet"

    if cache.exists() and not force:
        logger.info("Loading cached Meteostat parquet from %s", cache)
        return pd.read_parquet(cache)

    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end) if end else pd.Timestamp.now()

    logger.info("Fetching Meteostat for %d stations (%s → %s)…",
                len(DE_STATIONS), start_ts.date(), end_ts.date())

    station_dfs = []
    for name, station_id, lat, lon, alt in DE_STATIONS:
        logger.info("  %s (WMO %s)", name, station_id)
        df = _fetch_station_v2(station_id, name, start_ts, end_ts)
        if not df.empty:
            station_dfs.append(df)

    if not station_dfs:
        raise RuntimeError(
            "No weather data retrieved. Check your internet connection "
            "and that meteostat is installed: pip install meteostat"
        )

    merged = pd.concat(station_dfs, axis=1).sort_index()

    # National composite: mean across stations per variable
    for var in KEEP_COLS:
        cols = [c for c in merged.columns if c.endswith(f"_{var}")]
        if cols:
            merged[f"de_{var}"] = merged[cols].mean(axis=1).astype("float32")

    # Forward-fill short gaps (≤ 2 h) — common in station data
    merged = merged.ffill(limit=2)

    logger.info("Meteostat loaded: %d rows, %s → %s",
                len(merged), merged.index.min().date(), merged.index.max().date())

    merged.to_parquet(cache)
    logger.info("Cached Meteostat parquet → %s", cache)
    return merged
