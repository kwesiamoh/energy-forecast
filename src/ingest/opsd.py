"""
OPSD (Open Power System Data) ingestion module.

Downloads the time_series package from the OPSD data portal:
  https://data.open-power-system-data.org/time_series/

Primary dataset for this project (DE columns only):
  - German national load (actual consumption)
  - Solar PV generation (actual)
  - Wind onshore + offshore generation (actual)
  - Biomass, run-of-river, pumped storage, other renewables (actual)

Resolution: hourly
License: MIT / CC BY 4.0 — attribution required.

NOTE: OPSD stopped being updated after ~October 2020.  Data after that date
is backfilled from SMARD in the merge step (merge.py).  This module is
intentionally kept clean — it only downloads and parses historical OPSD data.
The SMARD backfill is the responsibility of merge.py (separation of concerns).
"""

import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/latest/"
    "time_series_60min_singleindex.csv"
)

# Columns to skip even if they match the DE_ prefix filter.
# We keep only *actual* generation/load, not forecasts, capacities, profiles.
_SKIP_SUBSTRINGS = ("forecast", "capacity", "profile")


def _normalize_opsd_column(raw_col: str) -> str | None:
    """
    Convert an OPSD raw column name to a clean project name.

    Rules (applied in order):
      1. timestamp → timestamp
      2. Non-DE columns → None (drop)
      3. Columns containing forecast / capacity / profile → None (drop)
      4. DE_load_actual_entsoe_transparency → load_mw
      5. DE_*_load_actual_* → *_load_mw
      6. DE_*_generation_actual → *_mw   (covers ALL fuel types dynamically)
      7. Everything else → None (drop)

    This version dynamically accepts any actual generation column regardless
    of fuel type — no hard allowlist that would silently drop biomass,
    run_of_river, pumped_storage, or other_renewables.
    """
    if raw_col == "utc_timestamp":
        return "timestamp"

    if not raw_col.startswith("DE_"):
        return None

    if any(skip in raw_col for skip in _SKIP_SUBSTRINGS):
        return None

    # Primary load
    if raw_col == "DE_load_actual_entsoe_transparency":
        return "load_mw"

    # Regional load (e.g. DE_50hertz_load_actual_entsoe_transparency)
    if "_load_actual" in raw_col:
        stem = raw_col[3:]  # remove "DE_"
        return stem.split("_load_actual")[0].lower() + "_load_mw"

    # Actual generation — accepts ALL fuel types dynamically
    if raw_col.endswith("_generation_actual"):
        # e.g. DE_solar_generation_actual             → solar_mw
        # e.g. DE_biomass_generation_actual           → biomass_mw
        # e.g. DE_run_of_river_and_poundage_generation_actual
        #                                             → run_of_river_and_poundage_mw
        stem = raw_col[3:]  # remove "DE_"
        return stem.replace("_generation_actual", "_mw").lower()

    return None


def _build_opsd_column_map(raw_path: Path) -> dict[str, str]:
    """Build a mapping from raw OPSD columns → clean project column names."""
    cols = pd.read_csv(raw_path, nrows=0).columns.tolist()
    mapping = {}
    for col in cols:
        normalized = _normalize_opsd_column(col)
        if normalized is not None:
            mapping[col] = normalized
    return mapping


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
        logger.info("OPSD raw file already exists at %s — skipping download.", dest)
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
      2. Select German columns only (all actual generation types — dynamic).
      3. Parse UTC timestamps, set as DatetimeIndex.
      4. Rename to short, clean column names.
      5. Drop rows where ALL generation columns are NaN.
      6. Convert to float32 and save parquet.

    Data coverage note:
      OPSD is authoritative through ~October 2020.  After that the columns
      go NaN.  SMARD backfill for those gaps is handled by merge.py —
      this module deliberately does NOT perform any backfill so that it
      remains a clean, single-responsibility loader.

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
    opsd_columns = _build_opsd_column_map(raw_path)
    cols_to_read = list(opsd_columns.keys())

    df = pd.read_csv(
        raw_path,
        usecols=cols_to_read,
        parse_dates=["utc_timestamp"],
        low_memory=False,
    )

    df = df.rename(columns=opsd_columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Drop rows where ALL generation signals are missing (sparse early years).
    gen_cols = [c for c in df.columns if c.endswith("_mw") and "_load_mw" not in c]
    if gen_cols:
        df = df.dropna(subset=gen_cols, how="all")

    # Convert MW columns → float32 to save memory
    for col in [c for c in df.columns if c.endswith("_mw")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    logger.info(
        "OPSD loaded: %d rows, %s → %s | columns: %s",
        len(df),
        df.index.min().date(),
        df.index.max().date(),
        sorted(df.columns.tolist()),
    )

    df.to_parquet(cache)
    logger.info("Cached OPSD parquet → %s", cache)
    return df
