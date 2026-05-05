"""
OPSD (Open Power System Data) ingestion module.

Downloads the time_series package from the OPSD data portal:
  https://data.open-power-system-data.org/time_series/

Primary dataset for this project:
  - German national load (actual consumption)
  - Solar PV generation (actual)
  - Wind onshore + offshore generation (actual)

Resolution: hourly (DE_load_actual_entsoe_transparency, etc.)
License: MIT / CC BY 4.0 — attribution required.
"""

import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Latest stable OPSD time-series release
OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/latest/"
    "time_series_60min_singleindex.csv"
)

# Columns we care about (DE = Germany)
OPSD_COLUMNS = {
    "utc_timestamp": "timestamp",
    "DE_load_actual_entsoe_transparency": "load_mw",
    "DE_solar_generation_actual": "solar_mw",
    "DE_wind_onshore_generation_actual": "wind_onshore_mw",
    "DE_wind_offshore_generation_actual": "wind_offshore_mw",
}


def download_opsd(raw_dir: Path, force: bool = False) -> Path:
    """
    Download the OPSD 60-min single-index CSV to raw_dir.

    Args:
        raw_dir:  Directory to save the raw file.
        force:    Re-download even if the file already exists.

    Returns:
        Path to the downloaded file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "opsd_time_series_60min.csv"

    if dest.exists() and not force:
        logger.info(
            "OPSD raw file already exists at %s — skipping download.", dest)
        return dest

    logger.info("Downloading OPSD time series from %s …", OPSD_URL)
    response = requests.get(OPSD_URL, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="OPSD"
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
            bar.update(len(chunk))

    logger.info("Saved OPSD raw file → %s", dest)
    return dest


def load_opsd(raw_dir: Path, processed_dir: Path, force: bool = False) -> pd.DataFrame:
    """
    Load and pre-process the OPSD dataset.

    Steps:
      1. Download if needed.
      2. Select German columns only.
      3. Parse UTC timestamps, set as DatetimeIndex.
      4. Rename to short, clean column names.
      5. Drop rows where ALL generation columns are NaN (sparse early years).
      6. Save processed parquet to processed_dir.

    Returns:
        Hourly DataFrame indexed by UTC timestamp.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / "opsd.parquet"

    if cache.exists() and not force:
        logger.info("Loading cached OPSD parquet from %s", cache)
        return pd.read_parquet(cache)

    raw_path = download_opsd(raw_dir, force=force)

    logger.info("Parsing OPSD CSV (this may take ~30 s for the full dataset) …")
    cols_to_read = list(OPSD_COLUMNS.keys())

    df = pd.read_csv(
        raw_path,
        usecols=cols_to_read,
        parse_dates=["utc_timestamp"],
        low_memory=False,
    )

    df = df.rename(columns=OPSD_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Drop rows where all generation signals are missing
    gen_cols = ["solar_mw", "wind_onshore_mw", "wind_offshore_mw"]
    df = df.dropna(subset=gen_cols, how="all")

    # Convert MW → float32 to save memory
    for col in ["load_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    logger.info(
        "OPSD loaded: %d rows, %s → %s",
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )

    df.to_parquet(cache)
    logger.info("Cached OPSD parquet → %s", cache)
    return df
