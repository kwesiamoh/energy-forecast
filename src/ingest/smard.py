"""
SMARD (Strommarktdaten) ingestion module.

SMARD is operated by the Bundesnetzagentur (Federal Network Agency, Germany).
It provides official real-time and historical electricity market data sourced
directly from TSOs (50Hertz, Amprion, TenneT, TransnetBW).

Data portal: https://www.smard.de/en
License: Data Licence Germany – Attribution – Version 2.0 (dl-de/by-2-0)
Attribution: © Bundesnetzagentur | SMARD.de

Used in this project as the VALIDATION dataset:
  - Cross-check OPSD load/generation figures
  - Fill gaps in OPSD where available

API:
  SMARD exposes a JSON API. Each "filter" ID maps to a series:
    410  = Actual generation: Photovoltaics [MWh]
    4066 = Actual generation: Wind onshore [MWh]
    4067 = Actual generation: Wind offshore [MWh]
    4359 = Actual consumption: Total [MWh]

  Timestamps are UNIX milliseconds, UTC.
  Data is returned in chunks of up to ~1 week at hourly resolution.

NOTE: The SMARD API is rate-limited. This module caches all responses
to avoid redundant requests. Use force=False (default) in production.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SMARD_BASE = "https://www.smard.de/app/chart_data"

# Filter IDs → clean column names
SMARD_FILTERS = {
    4359: "load_mwh",
    410:  "solar_mwh",
    4066: "wind_onshore_mwh",
    4067: "wind_offshore_mwh",
}

# SMARD resolution codes
RESOLUTION_HOUR = "hour"

# Throttle between requests (seconds) — be polite to the API
REQUEST_DELAY = 0.3


def _fetch_index(filter_id: int, resolution: str = RESOLUTION_HOUR) -> list[int]:
    """Fetch the list of available timestamp indices for a filter."""
    url = f"{SMARD_BASE}/{filter_id}/{filter_id}/{resolution}/index.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["timestamps"]


def _fetch_chunk(
    filter_id: int, timestamp_ms: int, resolution: str = RESOLUTION_HOUR
) -> pd.Series:
    """
    Fetch one data chunk for a filter starting at timestamp_ms.

    Returns a Series indexed by UTC DatetimeIndex with float values in MWh.
    """
    url = (
        f"{SMARD_BASE}/{filter_id}/{filter_id}/"
        f"{resolution}/{timestamp_ms}/data.json"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    series_data = resp.json()["series"]

    records = {}
    for ts_ms, value in series_data:
        if value is not None:
            ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
            records[ts] = float(value)

    return pd.Series(records, dtype="float32")


def download_smard_series(
    filter_id: int,
    raw_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.Series:
    """
    Download a full SMARD time series for a given filter_id.

    Results are cached as JSON in raw_dir to avoid re-downloading.

    Args:
        filter_id:  SMARD filter ID (see SMARD_FILTERS).
        raw_dir:    Directory for raw cache files.
        start:      ISO date string — skip chunks before this date.
        end:        ISO date string — skip chunks after this date (None = today).
        force:      Ignore cache and re-download everything.

    Returns:
        Hourly pd.Series indexed by UTC DatetimeIndex.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_file = raw_dir / f"smard_{filter_id}.json"

    if cache_file.exists() and not force:
        logger.info("Loading cached SMARD series %d from %s",
                    filter_id, cache_file)
        with open(cache_file) as f:
            data = json.load(f)
        series = pd.Series(
            {pd.Timestamp(k): v for k, v in data.items()}, dtype="float32"
        )
        series.index = pd.DatetimeIndex(series.index, tz="UTC")
        return series.sort_index()

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()

    logger.info(
        "Fetching SMARD filter %d (%s → %s) …",
        filter_id,
        start_ts.date(),
        end_ts.date(),
    )

    indices = _fetch_index(filter_id)
    chunks = []

    for ts_ms in indices:
        ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
        if ts < start_ts or ts > end_ts:
            continue
        logger.debug("  chunk %s", ts)
        chunk = _fetch_chunk(filter_id, ts_ms)
        if not chunk.empty:
            chunks.append(chunk)
        time.sleep(REQUEST_DELAY)

    if not chunks:
        logger.warning("No data returned for SMARD filter %d", filter_id)
        return pd.Series(dtype="float32")

    series = pd.concat(chunks).sort_index()
    series = series[~series.index.duplicated(keep="first")]

    # Cache to JSON
    cache_data = {str(k): float(v) for k, v in series.items()}
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)
    logger.info("Cached SMARD filter %d → %s", filter_id, cache_file)

    return series


def load_smard(
    raw_dir: Path,
    processed_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load all SMARD series and return a merged hourly DataFrame.

    The SMARD series use MWh (energy per hour) whereas OPSD uses MW (power).
    At hourly resolution these are numerically equivalent, but we keep the
    original SMARD units here and note the difference in column names.

    Returns:
        Hourly DataFrame indexed by UTC timestamp.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / "smard.parquet"

    if cache.exists() and not force:
        logger.info("Loading cached SMARD parquet from %s", cache)
        return pd.read_parquet(cache)

    all_series = {}
    for filter_id, col_name in SMARD_FILTERS.items():
        s = download_smard_series(
            filter_id, raw_dir, start=start, end=end, force=force
        )
        all_series[col_name] = s

    df = pd.DataFrame(all_series)
    df.index.name = "timestamp"
    df = df.sort_index()

    logger.info(
        "SMARD loaded: %d rows, %s → %s",
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )

    df.to_parquet(cache)
    logger.info("Cached SMARD parquet → %s", cache)
    return df
