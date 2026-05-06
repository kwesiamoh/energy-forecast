"""
Meteostat weather ingestion — 20-station spatially-weighted German grid.

Why 20 stations?
  Germany has distinct climate zones that matter for energy forecasting:
  - North Sea coast: high wind, low solar, maritime mild winters
  - Baltic coast: cold winters, good wind
  - Northwest lowlands (NRW): moderate, high population/load density
  - Central uplands (Harz, Eifel, Sauerland): orographic wind enhancement
  - Rhine valley: warmest region, highest solar yield
  - Bavaria/Alps foothills: cold winters, strong foehn events, high solar
  - East Germany (Brandenburg, Saxony): continental, high solar, some wind

Station selection criteria:
  1. Good hourly data availability back to 2015 (verified against Meteostat bulk)
  2. Geographic spread across all 16 Bundeslaender
  3. Co-located or near major renewable energy zones where possible
  4. Airport stations preferred (most complete records, longest history)

Spatial weighting:
  Rather than a simple mean, stations are weighted by the approximate area
  of Germany they represent (inverse-distance-squared Voronoi proxy).
  This prevents the dense cluster of southern stations from over-representing
  Bavaria relative to the sparsely-instrumented northeast.

Composite columns produced (prefix de_):
  de_temp    de_dwpt    de_rhum    de_prcp
  de_wspd    de_wpgt    de_pres    de_tsun

Per-station columns are also retained (e.g. hamburg_temp) for spatial
analysis, ablation studies, and regional sub-models.

Attribution: Weather data by Meteostat (https://meteostat.net) CC BY-NC 4.0
Install:     pip install --upgrade meteostat
"""

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Station grid ──────────────────────────────────────────────────────────────
# (name, WMO_ID, lat, lon, alt_m, region_description)
DE_STATIONS = [
    # ── North Sea coast & Schleswig-Holstein ──────────────────────────────
    ("sylt",         "10018", 54.91,  8.33,  26,
     "North Sea island, max wind exposure"),
    ("schleswig",    "10035", 54.53,  9.55,  43, "Schleswig-Holstein interior"),
    ("hamburg",      "10147", 53.63,  9.99,
     15, "Hamburg metro, port load centre"),
    # ── Baltic coast & Mecklenburg ────────────────────────────────────────
    ("rostock",      "10170", 54.18, 12.08,
     16, "Baltic coast, offshore wind hub"),
    ("greifswald",   "10184", 54.10, 13.40,
     20, "Northeast Baltic, wind corridor"),
    # ── Brandenburg & Berlin ──────────────────────────────────────────────
    ("berlin",       "10384", 52.47, 13.40,  50, "Berlin metro, continental east"),
    ("cottbus",      "10393", 51.78, 14.32,  69,
     "Lusatia, high solar, brown-coal legacy"),
    # ── Lower Saxony & Bremen ─────────────────────────────────────────────
    ("bremerhaven",  "10107", 53.52,  8.58,
     4, "North Sea estuary, offshore wind"),
    ("hannover",     "10338", 52.47,  9.70,  56, "Central Lower Saxony"),
    # ── North Rhine-Westphalia ────────────────────────────────────────────
    ("essen",        "10410", 51.40,  6.97,  60,
     "Ruhr metro, highest load density DE"),
    ("koeln",        "10513", 50.87,  7.17,  92, "Cologne / Rhine lowlands"),
    # ── Hesse & Thuringia ────────────────────────────────────────────────
    ("frankfurt",    "10637", 50.05,  8.60, 111, "Main economic hub, central DE"),
    ("erfurt",       "10452", 50.98, 10.97,
     316, "Thuringia basin, central uplands"),
    # ── Saxony ───────────────────────────────────────────────────────────
    ("dresden",      "10488", 51.13, 13.75, 227,
     "Saxony, continental east, solar belt"),
    # ── Rhineland-Palatinate & Saarland ──────────────────────────────────
    ("trier",        "10609", 49.75,  6.66, 265, "Mosel valley, SW Germany"),
    # ── Baden-Wuerttemberg ───────────────────────────────────────────────
    ("stuttgart",    "10739", 48.69,  9.22, 371,
     "Southwest, wine climate, high solar"),
    ("freiburg",     "10803", 47.99,  7.85, 237,
     "Upper Rhine, Germany's sunniest city"),
    # ── Bavaria ──────────────────────────────────────────────────────────
    ("nuernberg",    "10763", 49.50, 11.05, 312, "Franconia, central Bavaria"),
    ("munich",       "10866", 48.35, 11.79, 447, "Munich metro, Alpine foothills"),
    ("zugspitze",    "10961", 47.42, 10.98,
     2960, "Alpine high altitude, foehn signal"),
]

# Columns we want from the bulk CSV. Selected by name after reading,
# so we are resilient to schema changes or extra columns.
KEEP_COLS = ["temp", "dwpt", "rhum", "prcp", "wspd", "wpgt", "pres", "tsun"]

# ── Spatial weights (Voronoi-area proxy) ─────────────────────────────────────
STATION_WEIGHTS = {
    "sylt":        0.030,
    "schleswig":   0.055,
    "hamburg":     0.048,
    "rostock":     0.052,
    "greifswald":  0.045,
    "berlin":      0.062,
    "cottbus":     0.058,
    "bremerhaven": 0.038,
    "hannover":    0.065,
    "essen":       0.055,
    "koeln":       0.048,
    "frankfurt":   0.058,
    "erfurt":      0.052,
    "dresden":     0.055,
    "trier":       0.038,
    "stuttgart":   0.050,
    "freiburg":    0.032,
    "nuernberg":   0.055,
    "munich":      0.045,
    "zugspitze":   0.015,
}
_total = sum(STATION_WEIGHTS.values())
STATION_WEIGHTS = {k: v / _total for k, v in STATION_WEIGHTS.items()}


def _fetch_bulk_year(station_id: str, year: int) -> pd.DataFrame | None:
    """
    Download one year of hourly data from Meteostat's bulk CDN.

    The bulk CSV has a real header row whose exact column count varies by
    station and year (currently 24-26 columns). We let pandas read it
    normally and select whichever KEEP_COLS are present — this makes the
    function resilient to future schema additions or removals.

    Timestamp construction:
      Bulk files always include 'date' and 'hour' columns.
      We assemble a naive local (Europe/Berlin) timestamp, then convert to UTC.
    """
    url = f"https://data.meteostat.net/hourly/{year}/{station_id}.csv.gz"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        # Read normally — the file HAS a header row; do NOT pass header=None
        df = pd.read_csv(
            io.BytesIO(resp.content),
            compression="gzip",
            low_memory=False,
        )

        # Normalise column names (lowercase, strip whitespace)
        df.columns = [c.strip().lower() for c in df.columns]

        logger.debug(
            "  bulk %s/%d: %d cols → %s",
            station_id, year, len(df.columns), df.columns.tolist()
        )

        if "hour" not in df.columns:
            raise ValueError(
                f"Expected an 'hour' column, got: {df.columns.tolist()}"
            )

        df["hour"] = df["hour"].fillna(0).astype(int)

        if "date" in df.columns:
            # Older schema: single 'date' string + 'hour' int
            df["timestamp"] = (
                pd.to_datetime(df["date"], errors="coerce")
                + pd.to_timedelta(df["hour"], unit="h")
            )
        elif all(c in df.columns for c in ("year", "month", "day")):
            # Current schema: separate year / month / day / hour columns
            df["timestamp"] = pd.to_datetime(
                dict(year=df["year"], month=df["month"],
                     day=df["day"], hour=df["hour"]),
                errors="coerce",
            )
        else:
            raise ValueError(
                f"Cannot build timestamp from columns: {df.columns.tolist()}"
            )
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")

        # Localise naive Berlin time → UTC
        df.index = df.index.tz_localize(
            "Europe/Berlin", ambiguous="NaT", nonexistent="NaT"
        ).tz_convert("UTC")

        # Keep only the columns we care about (whatever subset exists)
        available = [c for c in KEEP_COLS if c in df.columns]
        if not available:
            logger.warning(
                "  bulk %s/%d: none of KEEP_COLS found in %s",
                station_id, year, df.columns.tolist()
            )
            return None

        return df[available].astype("float32")

    except Exception as exc:
        logger.warning("  bulk %s/%d error: %s", station_id, year, exc)
        return None


def _fetch_station_meteostat_lib(
    station_id: str, name: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame | None:
    """
    Try the meteostat library first (faster, handles gaps).

    Uses the class-based Hourly API (meteostat >= 1.6):
        ms.Hourly(station_id, start, end)   ← correct
        ms.hourly(...)                       ← removed in 1.6, do not use
    """
    try:
        import meteostat as ms
        ts = ms.Hourly(station_id, start, end)
        df = ts.fetch()
        if df is None or df.empty:
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize(
                "Europe/Berlin", ambiguous="NaT", nonexistent="NaT"
            )
        df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"
        available = [c for c in KEEP_COLS if c in df.columns]
        return df[available].astype("float32") if available else None
    except Exception as exc:
        logger.debug(
            "  meteostat lib failed for %s: %s — falling back to bulk", name, exc
        )
        return None


def _fetch_station(
    name: str,
    station_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch hourly data for one station.

    Strategy:
      1. Try meteostat library (class-based Hourly API) — fast, gap-filled
      2. Fall back to year-by-year bulk CSV download from Meteostat CDN
    """
    df = _fetch_station_meteostat_lib(station_id, name, start, end)
    if df is not None and not df.empty:
        logger.info("  %-14s (%s): %d rows via meteostat lib",
                    name, station_id, len(df))
        return df

    logger.info("  %-14s (%s): fetching bulk CSVs ...", name, station_id)
    years = range(start.year, end.year + 1)
    frames = [_fetch_bulk_year(station_id, yr) for yr in years]
    frames = [f for f in frames if f is not None]

    if not frames:
        logger.warning("  %-14s (%s): no data available", name, station_id)
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    # Drop NaT rows introduced by DST ambiguity before filtering
    df = df[df.index.notna()]
    df = df[(df.index >= start) & (df.index <= end)]
    df = df[~df.index.duplicated(keep="first")]
    logger.info("  %-14s (%s): %d rows via bulk CSV",
                name, station_id, len(df))
    return df


def _weighted_composite(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute area-weighted national composite columns (prefix de_).

    Missing stations at a given timestep are excluded from the denominator
    rather than pulling the weighted mean toward zero.
    """
    for var in KEEP_COLS:
        station_cols = [c for c in merged.columns if c.endswith(f"_{var}")]
        if not station_cols:
            continue

        weights = np.array([
            STATION_WEIGHTS.get(c.replace(f"_{var}", ""), 1.0)
            for c in station_cols
        ], dtype=np.float64)

        data = merged[station_cols].values.astype(np.float64)
        w_mat = np.where(np.isfinite(data), weights[None, :], 0.0)
        w_sum = w_mat.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum == 0, np.nan, w_sum)
        composite = (data * w_mat).sum(axis=1) / w_sum.squeeze()

        merged[f"de_{var}"] = composite.astype("float32")

    return merged


def load_meteostat(
    processed_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
    stations: list | None = None,
) -> pd.DataFrame:
    """
    Fetch hourly weather for the 20-station German grid and return a
    DataFrame with per-station columns + area-weighted national composites.

    Args:
        processed_dir: Where to cache meteostat.parquet.
        start:         ISO date string.
        end:           ISO date string (None = today).
        force:         Ignore cache and re-download.
        stations:      Override station list (default = DE_STATIONS).

    Returns:
        Hourly DataFrame indexed by UTC DatetimeIndex.
        Composite columns: de_temp, de_wspd, de_tsun, de_prcp, ...
        Per-station: hamburg_temp, munich_wspd, ...
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / "meteostat.parquet"

    if cache.exists() and not force:
        logger.info("Loading cached Meteostat parquet from %s", cache)
        return pd.read_parquet(cache)

    station_list = stations or DE_STATIONS
    start_ts = pd.Timestamp(start).tz_localize("UTC")
    end_ts = (
        pd.Timestamp(end).tz_localize("UTC") if end
        else pd.Timestamp.now(tz="UTC")
    )

    logger.info(
        "Fetching Meteostat for %d stations (%s -> %s) ...",
        len(station_list), start_ts.date(), end_ts.date(),
    )

    station_dfs = []
    for entry in station_list:
        name, station_id = entry[0], entry[1]
        df = _fetch_station(name, station_id, start_ts, end_ts)
        if not df.empty:
            df.columns = [f"{name}_{c}" for c in df.columns]
            station_dfs.append(df)

    if not station_dfs:
        raise RuntimeError(
            "No weather data retrieved from any station.\n"
            "Check: pip install --upgrade meteostat  and internet connectivity."
        )

    merged = pd.concat(station_dfs, axis=1).sort_index()
    merged = merged.ffill(limit=2)
    merged = _weighted_composite(merged)

    n_stations = len(station_dfs)
    n_composite = sum(1 for c in merged.columns if c.startswith("de_"))
    logger.info(
        "Meteostat loaded: %d rows, %d stations, %d composite columns | %s -> %s",
        len(merged), n_stations, n_composite,
        merged.index.min().date(), merged.index.max().date(),
    )

    merged.to_parquet(cache)
    logger.info("Cached Meteostat parquet -> %s", cache)
    return merged
