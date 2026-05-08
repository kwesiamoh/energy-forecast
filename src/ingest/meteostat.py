"""
Meteostat weather ingestion module.

Downloads hourly weather observations for 20 representative German stations
via the Meteostat Python library (v2.x).

⚠️  REQUIRES meteostat >= 2.0.0
    The library's API changed completely in v2.0.0 (released Dec 30, 2025).
    The old `from meteostat import Hourly; Hourly(id, start, end).fetch()` API
    was removed.  The new API is:
        import meteostat as ms
        ms.hourly(ms.Station(id='10382'), start, end).fetch()
    The legacy bulk.meteostat.net endpoint was also deprecated Jan 2026.
    Install/upgrade with: pip install "meteostat>=2.0.0"

Composite national features are computed as population-weighted means
across stations using approximate Voronoi area weights.

BUG FIX (v2) — de_tsun NaN 2018-2022:
  Meteostat bulk CSVs for many German stations are missing sunshine (tsun)
  data before mid-2022.  This module implements a two-stage fallback:
    1. Try Meteostat (existing logic, unchanged).
    2. For any station where tsun is >80% NaN, attempt to backfill from the
       DWD (Deutscher Wetterdienst) Open Data API — specifically the
       "stundenwerte_SD" (hourly sunshine duration) dataset.
    3. If DWD data is also unavailable AND the global DROP_TSUN_IF_SPARSE
       flag is True, de_tsun is dropped from the output and a WARNING is
       logged so downstream code (weather.py, pipeline.py) degrades
       gracefully.

Configuration:
  DROP_TSUN_IF_SPARSE (module-level global, default False):
    Set True to skip the DWD fallback entirely and simply omit de_tsun when
    it cannot be backfilled.  This is the safe option if you do not need the
    clearsky_index feature.  Do NOT pass this as a function argument — it is
    a module-level configuration variable.

Usage:
    from src.ingest.meteostat import load_meteostat, DROP_TSUN_IF_SPARSE
    # Optionally override before calling:
    # import src.ingest.meteostat as met_mod; met_mod.DROP_TSUN_IF_SPARSE = True
    weather = load_meteostat(processed_dir, start="2018-01-01")
"""

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Configuration (module-level globals — do not pass as function arguments) ──

# If True, drop de_tsun when it is >TSUN_MAX_NAN_FRAC missing rather than
# attempting the DWD fallback.  Set False (default) to attempt DWD backfill.
# Override this at import time if needed:
#   import src.ingest.meteostat as met_mod
#   met_mod.DROP_TSUN_IF_SPARSE = True
DROP_TSUN_IF_SPARSE: bool = False

# Fraction threshold above which a station's tsun column is considered sparse.
TSUN_MAX_NAN_FRAC: float = 0.80

# meteostat v2 raises ValueError for requests > 3 years by default.
# Our pipeline requests 7+ years (START="2018-01-01" → today), so we disable
# the guard here. The library handles chunking internally; this flag only
# blocks the request without providing any real protection.
try:
    import meteostat as _ms
    _ms.config.block_large_requests = False
except Exception:
    pass  # not yet installed — will fail with a clear message when first used

# German stations: (meteostat_id, dwd_station_id, name, lat, lon, area_weight)
# Area weights are approximate Voronoi areas (km²) — normalised in code.
STATIONS = [
    ("10382", "00433", "Berlin",        52.47, 13.40, 29_475),
    ("10400", "01078", "Leipzig",       51.42, 12.24, 14_800),
    ("10410", "02564", "Dresden",       51.11, 13.76, 12_900),
    ("10501", "00691", "Bremen",        53.05,  8.80,  8_400),
    ("10513", "01981", "Hamburg",       53.63,  9.99, 14_200),
    ("10338", "01757", "Hannover",      52.47,  9.69, 20_100),  # EDDV — WMO 10338 (was wrong 10601)
    ("10637", "03032", "Kassel",        51.30,  9.45, 16_700),
    ("10708", "05705", "Cologne",       50.87,  6.10, 13_400),
    ("10724", "01327", "Dortmund",      51.52,  7.60,  9_800),
    ("10739", "02290", "Essen",         51.40,  6.97,  7_200),
    ("10763", "04169", "Frankfurt",     50.05,  8.60, 19_300),
    ("10803", "05792", "Stuttgart",     48.69,  9.22, 20_800),
    ("10836", "01270", "Munich",        48.35, 11.79, 29_700),
    ("10852", "04626", "Nuremberg",     49.49, 11.08, 22_500),
    ("10868", "01993", "Augsburg",      48.43, 10.92, 15_600),
    ("10929", "01001", "Zugspitze",     47.42, 10.98,  4_100),  # Alpine
    ("10200", "00044", "Flensburg",     54.77,  9.38,  6_300),
    ("10147", "03378", "Rostock",       54.18, 12.08, 13_100),
    ("10554", "03931", "Erfurt",        50.98, 10.96, 11_200),  # EDDE — WMO 10554 (was wrong 10338)
    ("10727", "04371", "Muenster",      51.95,  7.60, 10_800),
]

# Meteostat columns we care about — v2 API column names.
# v1.x had "wpgt" (wind gust peak); v2 removed it from default hourly output.
# "wdir" (wind direction) and "cldc" (cloud cover %) are new in v2.
METEOSTAT_COLS = ["temp", "rhum", "prcp", "wspd", "wdir", "pres", "tsun", "cldc"]

# DWD Open Data base URL for hourly sunshine
DWD_BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/sunshine/historical"


# ── DWD fallback ──────────────────────────────────────────────────────────────

def _fetch_dwd_sunshine(dwd_station_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """
    Fetch hourly sunshine duration (SD_SO, in minutes) from DWD Open Data.

    DWD publishes zip archives with a fixed naming convention.  We list the
    directory, find the right file for the station, download, and parse it.

    Returns:
        UTC-indexed pd.Series of sunshine minutes per hour, or None on failure.
    """
    # Zero-pad station ID to 5 digits
    sid = str(dwd_station_id).zfill(5)
    dir_url = f"{DWD_BASE}/"

    try:
        # Get directory listing
        resp = requests.get(dir_url, timeout=30)
        resp.raise_for_status()
        # Find the zip file for this station
        lines = resp.text.splitlines()
        zip_name = next(
            (line.split('"')[1] for line in lines if f"_{sid}_" in line and line.endswith('.zip">')),
            None,
        )
        if zip_name is None:
            logger.debug("DWD: no zip found for station %s", sid)
            return None

        zip_url = dir_url + zip_name
        logger.info("  DWD sunshine fallback: downloading %s", zip_url)
        resp = requests.get(zip_url, timeout=120)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            # Find the data file inside the zip
            data_file = next(
                (n for n in z.namelist() if n.startswith("produkt_sd_stunde_")),
                None,
            )
            if data_file is None:
                logger.debug("DWD: no data file in zip for station %s", sid)
                return None

            with z.open(data_file) as f:
                dwd_df = pd.read_csv(f, sep=";", encoding="latin-1")

        # Parse timestamp: "MESS_DATUM" = YYYYMMDDHH (local time CET/CEST)
        dwd_df.columns = dwd_df.columns.str.strip()
        dwd_df["timestamp"] = pd.to_datetime(
            dwd_df["MESS_DATUM"].astype(str), format="%Y%m%d%H"
        )
        # DWD timestamps are CET (UTC+1); convert to UTC
        dwd_df["timestamp"] = dwd_df["timestamp"].dt.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC")

        # SD_SO = sunshine duration in minutes
        col = next((c for c in dwd_df.columns if "SD_SO" in c), None)
        if col is None:
            return None

        series = (
            dwd_df.set_index("timestamp")[col]
            .replace(-999, np.nan)
            .astype("float32")
        )
        series.name = "tsun"
        # Clip to requested range
        return series.loc[start:end]

    except Exception as exc:
        logger.warning("DWD fallback failed for station %s: %s", sid, exc)
        return None


# ── Meteostat loader ──────────────────────────────────────────────────────────

def _check_meteostat_version() -> tuple[int, int]:
    """
    Return the (major, minor) version of the installed meteostat package.
    Raises ImportError if not installed, RuntimeError if version can't be parsed.
    """
    import meteostat  # noqa: PLC0415
    version_str = getattr(meteostat, "__version__", "0.0.0")
    parts = version_str.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return 0, 0


def _load_station_meteostat(
    meteostat_id: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame | None:
    """
    Download hourly data for one station via the meteostat library.

    Supports both API generations:
      v2.x (>= 2.0.0, released Dec 2025):
        import meteostat as ms
        ms.hourly(ms.Station(id='10382'), start, end).fetch()
      v1.x (<= 1.7.x):
        from meteostat import Hourly
        Hourly('10382', start, end).fetch()

    The legacy bulk.meteostat.net endpoint was deprecated Jan 2026.
    v2 uses data.meteostat.net instead and is preferred.

    To upgrade:  pip install --upgrade meteostat
    """
    import traceback  # noqa: PLC0415
    import meteostat as ms  # noqa: PLC0415

    major, minor = _check_meteostat_version()
    start_naive = start.to_pydatetime().replace(tzinfo=None)
    end_naive   = end.to_pydatetime().replace(tzinfo=None)

    try:
        if major >= 2:
            # ── v2 API ────────────────────────────────────────────────────
            station_data = ms.hourly(
                ms.Station(id=meteostat_id),
                start_naive,
                end_naive,
            )
        else:
            # ── v1 API ────────────────────────────────────────────────────
            logger.warning(
                "meteostat v%d.%d detected (< 2.0). "
                "The v1 bulk endpoint is deprecated (Jan 2026) and may return "
                "empty data. Upgrade with: pip install --upgrade meteostat",
                major, minor,
            )
            from meteostat import Hourly  # noqa: PLC0415
            station_data = Hourly(meteostat_id, start_naive, end_naive)

        df = station_data.fetch()
        if df is None or df.empty:
            logger.warning(
                "Meteostat returned empty DataFrame for station %s "
                "(version=%d.%d). Check that the station ID is valid and "
                "the date range has coverage.", meteostat_id, major, minor
            )
            return None

        # Ensure UTC-aware index
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"

        available = [c for c in METEOSTAT_COLS if c in df.columns]
        if not available:
            logger.warning(
                "Station %s: none of %s found in columns %s",
                meteostat_id, METEOSTAT_COLS, df.columns.tolist()
            )
            return None

        return df[available].astype("float32")

    except Exception as exc:
        logger.warning(
            "Meteostat fetch failed for station %s (meteostat v%d.%d): %s\n%s",
            meteostat_id, major, minor, exc, traceback.format_exc()
        )
        return None


def _backfill_tsun_dwd(
    station_df: pd.DataFrame,
    dwd_station_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    If station_df.tsun is too sparse, attempt to fill from DWD and merge.

    This function is called only when the module-level DROP_TSUN_IF_SPARSE
    is False (the default).  It is NOT called when the flag is True.

    Returns station_df (possibly with tsun backfilled).
    """
    if "tsun" not in station_df.columns:
        return station_df

    nan_frac = station_df["tsun"].isna().mean()
    if nan_frac <= TSUN_MAX_NAN_FRAC:
        return station_df  # good enough

    logger.info(
        "  tsun is %.0f%% NaN — attempting DWD backfill for station %s",
        nan_frac * 100, dwd_station_id,
    )
    dwd_series = _fetch_dwd_sunshine(dwd_station_id, start, end)
    if dwd_series is None or dwd_series.empty:
        logger.warning("  DWD backfill returned nothing for station %s", dwd_station_id)
        return station_df

    # Reindex DWD to match station_df index, then fill gaps
    dwd_aligned = dwd_series.reindex(station_df.index)
    station_df["tsun"] = station_df["tsun"].fillna(dwd_aligned)
    filled_nan = station_df["tsun"].isna().mean()
    logger.info(
        "  After DWD backfill: tsun NaN reduced to %.1f%%", filled_nan * 100
    )
    return station_df


# ── National composite ────────────────────────────────────────────────────────

def _build_composite(
    station_frames: list[pd.DataFrame],
    weights: list[float],
    full_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build population-weighted national composite from per-station DataFrames.

    A station's contribution to any given hour is weighted by its area proxy
    and whether it has a valid reading (missing-station-aware denominator).
    """
    # Reindex all frames to the full hourly index
    reindexed = [df.reindex(full_index) for df in station_frames]

    composite_cols = METEOSTAT_COLS
    result = {}

    for col in composite_cols:
        arrays = []
        w_arr  = []
        for df, w in zip(reindexed, weights):
            if col in df.columns:
                arrays.append(df[col].values)
                w_arr.append(w)

        if not arrays:
            continue

        data = np.stack(arrays, axis=1)  # shape (T, n_stations)
        wts  = np.array(w_arr)           # shape (n_stations,)

        # Per-row weighted mean, ignoring NaN stations
        valid   = ~np.isnan(data)
        w_sum   = (valid * wts).sum(axis=1)
        w_data  = np.where(valid, data * wts, 0.0).sum(axis=1)

        with np.errstate(invalid="ignore"):
            composite = np.where(w_sum > 0, w_data / w_sum, np.nan)

        result[f"de_{col}"] = composite.astype("float32")

    return pd.DataFrame(result, index=full_index)


# ── Public loader ─────────────────────────────────────────────────────────────

def load_meteostat(
    processed_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load and pre-process Meteostat weather data for all German stations.

    Configuration is read from module-level globals (not function arguments):
      DROP_TSUN_IF_SPARSE — set True to skip DWD sunshine backfill.
      TSUN_MAX_NAN_FRAC   — threshold above which station tsun is sparse.

    Steps:
      1. Download each station via meteostat library.
      2. For stations with >TSUN_MAX_NAN_FRAC missing tsun, attempt DWD
         backfill UNLESS DROP_TSUN_IF_SPARSE is True.
      3. Build population-weighted national composites (de_* columns).
      4. Build per-station DataFrame (berlin_temp, etc.) for spatial models.
      5. If de_tsun is still >TSUN_MAX_NAN_FRAC NaN after all fallbacks,
         drop it and log a WARNING so downstream code degrades gracefully.

    Args:
        processed_dir:  Directory for cached parquet output.
        start:          ISO date string for the start of the requested range.
        end:            ISO date string for the end (None = today).
        force:          Recompute even if cache exists.

    Returns:
        Hourly DataFrame indexed by UTC timestamp.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / "meteostat.parquet"

    if cache.exists() and not force:
        logger.info("Loading cached Meteostat parquet from %s", cache)
        return pd.read_parquet(cache)

    # Log the effective configuration so the user can see what will happen
    logger.info(
        "Meteostat config: DROP_TSUN_IF_SPARSE=%s, TSUN_MAX_NAN_FRAC=%.0f%%",
        DROP_TSUN_IF_SPARSE, TSUN_MAX_NAN_FRAC * 100,
    )

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()
    full_index = pd.date_range(start=start_ts, end=end_ts, freq="1h", tz="UTC")

    raw_weights = [s[5] for s in STATIONS]
    total_w     = sum(raw_weights)
    norm_weights = [w / total_w for w in raw_weights]

    station_frames: list[pd.DataFrame] = []
    station_names:  list[str]          = []

    for (met_id, dwd_id, name, lat, lon, _), w in zip(STATIONS, norm_weights):
        slug = name.lower().replace("ü", "ue").replace("ö", "oe")
        logger.info("Fetching station: %s (%s) …", name, met_id)

        sdf = _load_station_meteostat(met_id, start_ts, end_ts)
        if sdf is None or sdf.empty:
            logger.warning("  No data for %s — skipping.", name)
            continue

        # Attempt DWD sunshine backfill if the module flag permits it.
        # This is controlled by the module-level DROP_TSUN_IF_SPARSE global,
        # NOT by a function argument (removed to simplify the public API).
        if not DROP_TSUN_IF_SPARSE:
            sdf = _backfill_tsun_dwd(sdf, dwd_id, start_ts, end_ts)
        else:
            logger.debug(
                "  DWD tsun backfill skipped for %s (DROP_TSUN_IF_SPARSE=True)", name
            )

        station_frames.append(sdf)
        station_names.append(slug)

    if not station_frames:
        try:
            major, minor = _check_meteostat_version()
            version_info = f"meteostat v{major}.{minor} is installed"
            if major < 2:
                version_info += (
                    " — this is the OLD v1 API whose bulk endpoint was "
                    "deprecated Jan 2026. Upgrade: pip install --upgrade meteostat"
                )
        except ImportError:
            version_info = "meteostat is NOT installed. Install: pip install meteostat"
        raise RuntimeError(
            f"No Meteostat station data could be loaded. {version_info}. "
            "Check the WARNING log lines above for the per-station errors."
        )

    # Build national composites
    composite = _build_composite(station_frames, norm_weights[:len(station_frames)], full_index)

    # Build per-station DataFrame (only for stations we loaded)
    per_station_parts = []
    for sdf, slug in zip(station_frames, station_names):
        sdf_ri = sdf.reindex(full_index)
        sdf_ri.columns = [f"{slug}_{c}" for c in sdf_ri.columns]
        per_station_parts.append(sdf_ri)

    per_station = pd.concat(per_station_parts, axis=1)
    result = pd.concat([composite, per_station], axis=1)
    result.index.name = "timestamp"

    # ── de_tsun quality gate ──────────────────────────────────────────────
    if "de_tsun" in result.columns:
        tsun_nan = result["de_tsun"].isna().mean()
        if tsun_nan > TSUN_MAX_NAN_FRAC:
            logger.warning(
                "de_tsun is %.0f%% NaN after all fallbacks. "
                "Dropping it from the output. "
                "Set DROP_TSUN_IF_SPARSE=True (module global) to suppress "
                "the DWD attempt, or investigate DWD availability for your "
                "date range.",
                tsun_nan * 100,
            )
            result = result.drop(columns=["de_tsun"])
            # Also drop any per-station tsun columns
            tsun_cols = [c for c in result.columns if c.endswith("_tsun")]
            result = result.drop(columns=tsun_cols, errors="ignore")

    logger.info(
        "Meteostat loaded: %d rows, %s → %s | composite columns: %s",
        len(result),
        result.index.min().date(),
        result.index.max().date(),
        [c for c in result.columns if c.startswith("de_")],
    )

    result.to_parquet(cache)
    logger.info("Cached Meteostat parquet → %s", cache)
    return result
