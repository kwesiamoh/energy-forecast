"""
SMARD (Strommarktdaten) ingestion module.

SMARD is operated by the Bundesnetzagentur (Federal Network Agency, Germany).
It provides official real-time and historical electricity market data sourced
directly from TSOs (50Hertz, Amprion, TenneT, TransnetBW).

Data portal: https://www.smard.de/en
License: Data Licence Germany – Attribution – Version 2.0 (dl-de/by-2-0)
Attribution: © Bundesnetzagentur | SMARD.de
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Filter IDs → clean column names
SMARD_FILTERS = {
    4359: "load_mwh",
    410:  "solar_mwh",
    4066: "wind_onshore_mwh",
    4067: "wind_offshore_mwh",
}

SMARD_COLUMN_KEYWORDS = {
    "solar_mwh": ["solar", "photovoltaic", "pv"],
    "wind_onshore_mwh": ["wind onshore", "onshore"],
    "wind_offshore_mwh": ["wind offshore", "offshore"],
    "load_mwh": ["load", "consumption", "consumed", "demand"],
}

SMARD_SOURCE_PREFIXES = [
    "smard_",
    "actual_",
]


def download_smard_series(
    filter_id: int,
    raw_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.Series:
    """
    Download a full SMARD time series for a given filter_id.
    🚨 SMARD API DISCONTINUED - MANUAL DOWNLOAD REQUIRED 🚨
    """
    logger.warning(
        "SMARD API discontinued: Cannot download data automatically. "
        "Please download manually from https://www.smard.de/en/downloadcenter/download-market-data "
        "and place CSV files in %s", raw_dir
    )

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()
    empty_index = pd.date_range(start=start_ts, end=end_ts, freq="h", tz="UTC")

    return pd.Series(index=empty_index, dtype="float32")


def _read_smard_csv(path: Path) -> pd.DataFrame:
    """Read a SMARD CSV file and return a parsed DataFrame with UTC index."""
    # Efficient separator check
    with open(path, 'r', encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","

    df = pd.read_csv(
        path,
        sep=sep,
        parse_dates=[0],
        index_col=0,
        thousands=",",
        decimal='.',
        na_values=['-'],
        low_memory=False,
    )
    if df.empty:
        raise ValueError(f"SMARD CSV {path} is empty")

    if not pd.api.types.is_datetime64_any_dtype(df.index.dtype):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Correct Timezone Localization
    if df.index.tz is None:
        df.index = df.index.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if "end date" in (c.lower() for c in df.columns):
        df = df.drop(
            columns=[c for c in df.columns if c.lower() == "end date"])

    df = df.sort_index()
    return df


def _normalize_smard_column_name(name: str) -> str | None:
    """Map SMARD column or filename text to a target output column name."""
    s = str(name).strip().lower()
    s = s.replace("_", " ").replace("-", " ")

    for target, keywords in SMARD_COLUMN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in s:
                return target
    return None


def _extract_smard_series(df: pd.DataFrame, path: Path) -> dict[str, pd.Series]:
    """Extract known SMARD series from a DataFrame based on column names and file metadata."""
    series_map: dict[str, pd.Series] = {}
    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError(f"No numeric value column found in SMARD CSV {path}")

    for col in numeric_cols:
        target = _normalize_smard_column_name(col)
        if target:
            series_map[target] = df[col].astype("float32")

    if not series_map and len(numeric_cols) == 1:
        file_target = _normalize_smard_column_name(path.stem)
        if file_target:
            series_map[file_target] = df[numeric_cols[0]].astype("float32")

    return series_map


def _find_smard_files(raw_dir: Path) -> list[Path]:
    """Find manual SMARD CSV files that can be loaded by the pipeline."""
    files = []
    if not raw_dir.exists():
        return files

    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ".csv":
            continue
        name = path.name.lower()
        if any(name.startswith(prefix) for prefix in SMARD_SOURCE_PREFIXES):
            files.append(path)
    return sorted(files)


def load_smard(
    raw_dir: Path,
    processed_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load all SMARD series and return a merged hourly DataFrame.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / "smard.parquet"

    if cache.exists() and not force:
        logger.info("Loading cached SMARD parquet from %s", cache)
        return pd.read_parquet(cache)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()
    full_index = pd.date_range(start=start_ts, end=end_ts, freq="h", tz="UTC")

    loaded_any = False
    all_series: dict[str, pd.Series | None] = {
        col_name: None for col_name in SMARD_FILTERS.values()
    }

    files = _find_smard_files(raw_dir)
    if not files:
        logger.warning(
            "SMARD manual CSV files not found in %s. Looking for names like smard_*.csv or Actual_generation_*.csv.",
            raw_dir,
        )

    for path in files:
        try:
            df = _read_smard_csv(path)
            df = df.loc[start_ts:end_ts]
            extracted = _extract_smard_series(df, path)
            if not extracted:
                logger.warning(
                    "SMARD file %s did not contain known columns and was skipped.", path.name)
                continue

            for target, series in extracted.items():
                if target in all_series and all_series[target] is None:
                    # Align the series to the continuous hourly index immediately
                    all_series[target] = series.reindex(full_index)
                    loaded_any = True
                    logger.info("Loaded SMARD series %s from %s",
                                target, path.name)
                elif target in all_series:
                    logger.info(
                        "SMARD series %s already loaded; skipping duplicate from %s", target, path.name)
        except Exception as exc:
            logger.warning(
                "Failed to parse SMARD file %s: %s. Using empty series for its expected columns.", path.name, exc)

    # Fill any remaining unloaded series with empty NaNs mapped to the full index
    for col_name, series in all_series.items():
        if series is None:
            all_series[col_name] = pd.Series(index=full_index, dtype="float32")

    df = pd.DataFrame(all_series)
    df.index.name = "timestamp"
    df = df.sort_index()

    if loaded_any:
        logger.info("SMARD loaded from CSV: %d rows, %s → %s", len(
            df), df.index.min().date(), df.index.max().date())
    else:
        logger.warning(
            "SMARD loaded empty: no manual CSV files found. Manual download required.")

    df.to_parquet(cache)
    logger.info("Cached SMARD parquet → %s", cache)
    return df
