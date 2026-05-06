"""
SMARD (Strommarktdaten) ingestion module — with retry + resume support.

SMARD is operated by the Bundesnetzagentur (Federal Network Agency, Germany).
Data portal: https://www.smard.de/en
License: Data Licence Germany - Attribution - Version 2.0 (dl-de/by-2-0)
Attribution: Bundesnetzagentur | SMARD.de

Correct URL pattern (filter and region duplicated in filename -- by SMARD design):
  Index:  /chart_data/{filter}/{region}/index_{resolution}.json
  Data:   /chart_data/{filter}/{region}/{filter}_{region}_{resolution}_{timestamp}.json

Filter IDs for Germany (region="DE"):
  410   = Stromverbrauch: Gesamt (Netzlast)    -> load_mwh
  4068  = Stromerzeugung: Photovoltaik          -> solar_mwh
  4067  = Stromerzeugung: Wind Onshore          -> wind_onshore_mwh
  1225  = Stromerzeugung: Wind Offshore         -> wind_offshore_mwh

Robustness features:
  - HTTPAdapter with urllib3 Retry: auto-retries DNS failures, timeouts,
    429 rate-limits, and 5xx errors with exponential backoff.
  - Per-chunk progress cache: each fetched week is saved as a small JSON
    file in data/raw/smard_{filter_id}/. Interrupted runs resume from
    where they left off -- already-fetched chunks are never re-downloaded.
  - Failed chunks are logged and skipped; the rest of the series is
    still returned. Re-running with force=False fills gaps automatically.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

SMARD_BASE     = "https://www.smard.de/app/chart_data"
REGION         = "DE"
REQUEST_DELAY  = 1.0   # seconds between chunk requests -- be polite
MAX_RETRIES    = 5     # total attempts per chunk
BACKOFF_FACTOR = 2.0   # wait: 1s, 2s, 4s, 8s, 16s

SMARD_FILTERS = {
    410:  "load_mwh",
    4068: "solar_mwh",
    4067: "wind_onshore_mwh",
    1225: "wind_offshore_mwh",
}


def _make_session() -> requests.Session:
    """Session with automatic retry + exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


_SESSION = _make_session()


def _fetch_index(filter_id: int, resolution: str = "hour") -> list:
    url  = f"{SMARD_BASE}/{filter_id}/{REGION}/index_{resolution}.json"
    resp = _SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["timestamps"]


def _fetch_chunk(filter_id: int, timestamp_ms: int, resolution: str = "hour"):
    """
    Fetch one weekly chunk. Returns a pd.Series or None (on 404 / empty).
    """
    url = (
        f"{SMARD_BASE}/{filter_id}/{REGION}/"
        f"{filter_id}_{REGION}_{resolution}_{timestamp_ms}.json"
    )
    resp = _SESSION.get(url, timeout=30)

    if resp.status_code == 404:
        return None   # chunk not yet published -- not an error

    resp.raise_for_status()

    records = {}
    for ts_ms, value in resp.json().get("series", []):
        if value is not None:
            records[pd.Timestamp(ts_ms, unit="ms", tz="UTC")] = float(value)

    return pd.Series(records, dtype="float32") if records else None


def download_smard_series(
    filter_id,
    raw_dir,
    start="2015-01-01",
    end=None,
    force=False,
):
    """
    Download a full SMARD time series with per-chunk resume support.

    Cache layout:
        data/raw/smard_{filter_id}/          <- one JSON file per weekly chunk
        data/raw/smard_{filter_id}.json      <- final merged series (on completion)

    Re-running with force=False resumes from the last successful chunk
    and fills in any previously-failed gaps.
    """
    raw_dir      = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_cache  = raw_dir / f"smard_{filter_id}.json"
    chunk_dir    = raw_dir / f"smard_{filter_id}"

    # Fast path: already fully downloaded
    if final_cache.exists() and not force:
        logger.info("Loading cached SMARD %d from %s", filter_id, final_cache)
        with open(final_cache) as f:
            data = json.load(f)
        series = pd.Series(
            {pd.Timestamp(k): float(v) for k, v in data.items()},
            dtype="float32",
        )
        series.index = pd.DatetimeIndex(series.index, tz="UTC")
        return series.sort_index()

    chunk_dir.mkdir(exist_ok=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC") if end else pd.Timestamp.utcnow()

    logger.info(
        "Fetching SMARD filter %d (%s -> %s)...",
        filter_id, start_ts.date(), end_ts.date(),
    )

    indices  = _fetch_index(filter_id)
    in_range = [
        ts for ts in indices
        if start_ts <= pd.Timestamp(ts, unit="ms", tz="UTC") <= end_ts
    ]
    total  = len(in_range)
    failed = []

    for i, ts_ms in enumerate(in_range, 1):
        ts         = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
        chunk_file = chunk_dir / f"{ts_ms}.json"

        # Resume: skip already-saved chunks
        if chunk_file.exists() and not force:
            logger.debug("  [%d/%d] cached  %s", i, total, ts.date())
            continue

        logger.info("  [%d/%d] fetching %s ...", i, total, ts.date())
        try:
            chunk = _fetch_chunk(filter_id, ts_ms)
            if chunk is not None and not chunk.empty:
                with open(chunk_file, "w") as f:
                    json.dump({str(k): float(v) for k, v in chunk.items()}, f)
            else:
                logger.debug("  [%d/%d] %s: empty/404 -- skipped", i, total, ts.date())
        except Exception as exc:
            logger.warning("  [%d/%d] %s FAILED: %s", i, total, ts.date(), exc)
            failed.append(ts_ms)

        time.sleep(REQUEST_DELAY)

    if failed:
        logger.warning(
            "Filter %d: %d chunks failed after %d retries. "
            "Gaps will appear. Re-run with force=False to retry.",
            filter_id, len(failed), MAX_RETRIES,
        )

    # Merge all saved chunk files into one series
    all_records = {}
    for chunk_file in sorted(chunk_dir.glob("*.json")):
        with open(chunk_file) as f:
            all_records.update(json.load(f))

    if not all_records:
        logger.warning("No data collected for SMARD filter %d", filter_id)
        return pd.Series(dtype="float32")

    series = pd.Series(
        {pd.Timestamp(k): float(v) for k, v in all_records.items()},
        dtype="float32",
    )
    series.index = pd.DatetimeIndex(series.index, tz="UTC")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="first")]

    with open(final_cache, "w") as f:
        json.dump({str(k): float(v) for k, v in series.items()}, f)
    logger.info(
        "Cached SMARD %d -> %s  (%d rows, %d failed chunks)",
        filter_id, final_cache, len(series), len(failed),
    )
    return series


def load_smard(raw_dir, processed_dir, start="2015-01-01", end=None, force=False):
    """Load all SMARD filters and return a merged hourly DataFrame."""
    processed_dir = Path(processed_dir)
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
        "SMARD loaded: %d rows, %s -> %s",
        len(df), df.index.min().date(), df.index.max().date(),
    )
    df.to_parquet(cache)
    return df
