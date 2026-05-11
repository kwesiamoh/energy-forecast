"""
SMARD (Strommarktdaten) ingestion module — with retry + resume support.

SMARD is operated by the Bundesnetzagentur (Federal Network Agency, Germany).
Data portal: https://www.smard.de/en
License: Data Licence Germany - Attribution - Version 2.0 (dl-de/by-2-0)
Attribution: Bundesnetzagentur | SMARD.de

Correct URL pattern (filter and region duplicated in filename — by SMARD design):
  Index: /chart_data/{filter}/{region}/index_{resolution}.json
  Data:  /chart_data/{filter}/{region}/{filter}_{region}_{resolution}_{ts}.json

Filter IDs for Germany (region="DE") — verified against the official Bundesnetzagentur
  OpenAPI spec (https://github.com/bundesAPI/smard-api/blob/main/openapi.yaml).
  The spec's full filter enum is documented inline in SMARD_FILTERS below.

  ── Load ──────────────────────────────────────────────────────────────────
  410   = Stromverbrauch: Gesamt (Netzlast)           → load_mwh
  4359  = Stromverbrauch: Residuallast                → residual_load_mwh
  4387  = Stromverbrauch: Pumpspeicher                → pumped_storage_cons_mwh

  ── Renewables ────────────────────────────────────────────────────────────
  4068  = Stromerzeugung: Photovoltaik                → solar_mwh
  4067  = Stromerzeugung: Wind Onshore                → wind_onshore_mwh
  1225  = Stromerzeugung: Wind Offshore               → wind_offshore_mwh
  4066  = Stromerzeugung: Biomasse                    → biomass_mwh
  1226  = Stromerzeugung: Wasserkraft                 → run_of_river_mwh
  4070  = Stromerzeugung: Pumpspeicher                → pumped_storage_gen_mwh
  1228  = Stromerzeugung: Sonstige Erneuerbare        → other_renewables_mwh

  ── Conventional ──────────────────────────────────────────────────────────
  1224  = Stromerzeugung: Kernenergie                 → nuclear_mwh
  1223  = Stromerzeugung: Braunkohle                  → lignite_mwh
  4069  = Stromerzeugung: Steinkohle                  → hard_coal_mwh
  4071  = Stromerzeugung: Erdgas                      → gas_mwh
  1227  = Stromerzeugung: Sonstige Konventionelle     → other_conventional_mwh

  ── Market ────────────────────────────────────────────────────────────────
  4169  = Marktpreis: Deutschland/Luxemburg           → day_ahead_price_eur

DUAL ROLE — primary source and OPSD backfill:
  After ~October 2020, OPSD data becomes unavailable.  This module's output
  (load_mwh, solar_mwh, wind_onshore_mwh, wind_offshore_mwh) is used by
  merge.py to backfill the OPSD target columns for the 2020-present period.
  Unit note: SMARD data is MWh per hour.  At hourly resolution this is
  numerically equal to average MW — merge.py documents this convention.

Carbon intensity:
  carbon_intensity_g_kwh is computed after all generation series are loaded.
  Each fuel type is multiplied by a standard lifecycle emission factor
  (g CO₂-eq / kWh) and the sum is divided by total generation.  This gives
  an hourly grid carbon intensity signal useful as both a feature and a
  target for the carbon-aware forecasting component.

  Emission factors (g CO₂-eq / kWh, lifecycle median values):
    Lignite (brown coal) : 1000
    Hard coal            :  820
    Natural gas          :  490
    Biomass              :  230
    Solar PV             :   40
    Wind (on + offshore) :   15
    Run-of-river hydro   :   15
    Pumped storage gen   :   15  (proxy — hydro-like)
    Nuclear              :   12
    Other renewables     :   40  (proxy — solar-like)
    Other conventional   :  700  (proxy — coal/gas/oil mix; oil not in SMARD API)

Robustness features:
  - HTTPAdapter with urllib3 Retry: auto-retries DNS failures, timeouts,
    429 rate-limits, and 5xx errors with exponential backoff.
  - Per-chunk progress cache: each fetched week is saved as a small JSON
    file in data/raw/smard_{filter_id}/. Interrupted runs resume from
    where they left off — already-fetched chunks are never re-downloaded.
  - Failed chunks are logged and skipped; re-running with force=False
    automatically retries gaps.
  - Derived aggregate columns (total_generation, renewables, conventional,
    renewable_share, carbon_intensity_g_kwh) are computed at load time.
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
REQUEST_DELAY  = 1.0   # seconds between chunk requests — be polite
MAX_RETRIES    = 5
BACKOFF_FACTOR = 2.0   # waits: 1 s, 2 s, 4 s, 8 s, 16 s

# ── Filter ID → column name mapping ──────────────────────────────────────────
# All IDs verified against the official Bundesnetzagentur OpenAPI spec:
# https://github.com/bundesAPI/smard-api/blob/main/openapi.yaml
# The spec's complete filter enum is:
#   1223,1224,1225,1226,1227,1228,4066,4067,4068,4069,4070,4071,
#   410,4359,4387,4169,5078,4996,4997,4170,252–262,3791,123,126,715,5097,122
# Only IDs relevant to DE generation/load/price are included below.
SMARD_FILTERS: dict[int, str] = {
    # Load
    410:     "load_mwh",              # Stromverbrauch: Gesamt (Netzlast)
    4359:    "residual_load_mwh",     # Stromverbrauch: Residuallast
    4387:    "pumped_storage_cons_mwh",  # Stromverbrauch: Pumpspeicher
    # Renewables
    4068:    "solar_mwh",             # Stromerzeugung: Photovoltaik
    4067:    "wind_onshore_mwh",      # Stromerzeugung: Wind Onshore
    1225:    "wind_offshore_mwh",     # Stromerzeugung: Wind Offshore
    4066:    "biomass_mwh",           # Stromerzeugung: Biomasse
    1226:    "run_of_river_mwh",      # Stromerzeugung: Wasserkraft
    4070:    "pumped_storage_gen_mwh",# Stromerzeugung: Pumpspeicher
    1228:    "other_renewables_mwh",  # Stromerzeugung: Sonstige Erneuerbare
    # Conventional
    1224:    "nuclear_mwh",           # Stromerzeugung: Kernenergie (zero post Apr-2023)
    1223:    "lignite_mwh",           # Stromerzeugung: Braunkohle
    4069:    "hard_coal_mwh",         # Stromerzeugung: Steinkohle
    4071:    "gas_mwh",               # Stromerzeugung: Erdgas
    1227:    "other_conventional_mwh",# Stromerzeugung: Sonstige Konventionelle
    # Market price
    4169:    "day_ahead_price_eur",   # Marktpreis: Deutschland/Luxemburg
}

# Columns that make up each aggregate
_RENEWABLE_COLS = [
    "solar_mwh", "wind_onshore_mwh", "wind_offshore_mwh",
    "biomass_mwh", "run_of_river_mwh", "pumped_storage_gen_mwh",
    "other_renewables_mwh",
]
_CONVENTIONAL_COLS = [
    "nuclear_mwh", "lignite_mwh", "hard_coal_mwh",
    "gas_mwh", "other_conventional_mwh",
]

# ── Carbon intensity emission factors (g CO₂-eq / kWh, lifecycle medians) ────
# Sources: IPCC AR6, UBA Umweltbundesamt, EMBER 2023
_EMISSION_FACTORS_G_KWH: dict[str, float] = {
    "lignite_mwh":              1000.0,   # brown coal — highest carbon fuel
    "hard_coal_mwh":             820.0,
    "gas_mwh":                   490.0,   # natural gas CCGT
    "other_conventional_mwh":    700.0,   # proxy — coal/gas mix (incl. oil)
    "biomass_mwh":               230.0,   # lifecycle incl. land use
    "solar_mwh":                  40.0,   # lifecycle manufacturing
    "wind_onshore_mwh":           15.0,
    "wind_offshore_mwh":          15.0,
    "run_of_river_mwh":           15.0,
    "pumped_storage_gen_mwh":     15.0,   # proxy — hydro-like
    "nuclear_mwh":                12.0,
    "other_renewables_mwh":       40.0,   # proxy — solar-like
}


# ── Session factory ───────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Return a requests Session with automatic retry + exponential backoff."""
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


# ── Low-level API helpers ─────────────────────────────────────────────────────

def _fetch_index(filter_id: int, resolution: str = "hour") -> list[int]:
    url  = f"{SMARD_BASE}/{filter_id}/{REGION}/index_{resolution}.json"
    resp = _SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["timestamps"]


def _fetch_chunk(
    filter_id: int, timestamp_ms: int, resolution: str = "hour"
) -> pd.Series | None:
    """
    Fetch one weekly chunk.

    Returns:
        pd.Series (UTC-indexed) or None if the chunk is 404 / empty.
    """
    url = (
        f"{SMARD_BASE}/{filter_id}/{REGION}/"
        f"{filter_id}_{REGION}_{resolution}_{timestamp_ms}.json"
    )
    resp = _SESSION.get(url, timeout=30)

    if resp.status_code == 404:
        return None  # not yet published — not an error

    resp.raise_for_status()

    records: dict[pd.Timestamp, float] = {}
    for ts_ms, value in resp.json().get("series", []):
        if value is not None:
            records[pd.Timestamp(ts_ms, unit="ms", tz="UTC")] = float(value)

    return pd.Series(records, dtype="float32") if records else None


# ── Per-series downloader with chunk-level resume ─────────────────────────────

def download_smard_series(
    filter_id: int,
    raw_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.Series:
    """
    Download a full SMARD time series with per-chunk resume support.

    Cache layout:
        data/raw/smard_{filter_id}/     ← one JSON file per weekly chunk
        data/raw/smard_{filter_id}.json ← merged series (written on completion)

    Re-running with force=False resumes from the last successful chunk and
    fills in any previously-failed gaps automatically.
    """
    raw_dir      = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_cache  = raw_dir / f"smard_{filter_id}.json"
    chunk_dir    = raw_dir / f"smard_{filter_id}"

    # Fast path — already fully downloaded
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
    end_ts   = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()

    logger.info(
        "Fetching SMARD filter %d (%s → %s) …",
        filter_id, start_ts.date(), end_ts.date(),
    )

    indices  = _fetch_index(filter_id)
    in_range = [
        ts for ts in indices
        if start_ts <= pd.Timestamp(ts, unit="ms", tz="UTC") <= end_ts
    ]
    total  = len(in_range)
    failed: list[int] = []

    for i, ts_ms in enumerate(in_range, 1):
        ts         = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
        chunk_file = chunk_dir / f"{ts_ms}.json"

        if chunk_file.exists() and not force:
            logger.debug("  [%d/%d] cached  %s", i, total, ts.date())
            continue

        logger.info("  [%d/%d] fetching %s …", i, total, ts.date())
        try:
            chunk = _fetch_chunk(filter_id, ts_ms)
            if chunk is not None and not chunk.empty:
                with open(chunk_file, "w") as f:
                    json.dump({str(k): float(v) for k, v in chunk.items()}, f)
            else:
                logger.debug("  [%d/%d] %s: empty/404 — skipped", i, total, ts.date())
        except Exception as exc:
            logger.warning("  [%d/%d] %s FAILED: %s", i, total, ts.date(), exc)
            failed.append(ts_ms)

        time.sleep(REQUEST_DELAY)

    if failed:
        logger.warning(
            "Filter %d: %d chunks failed after %d retries. "
            "Re-run with force=False to retry gaps.",
            filter_id, len(failed), MAX_RETRIES,
        )

    # Merge all saved chunk files into one series
    all_records: dict[str, float] = {}
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
        "Cached SMARD %d → %s  (%d rows, %d failed chunks)",
        filter_id, final_cache, len(series), len(failed),
    )
    return series


# ── Carbon intensity calculation ──────────────────────────────────────────────

def _compute_carbon_intensity(df: pd.DataFrame) -> pd.Series:
    """
    Compute hourly grid carbon intensity in g CO₂-eq / kWh.

    Each generation source is weighted by its lifecycle emission factor and
    summed; the result is divided by total generation (MWh → kWh conversion
    cancels because both numerator and denominator use MWh units at the same
    scale — we want g/kWh so we adjust the denominator by ×1000).

    Formula:
        carbon_intensity = Σ(gen_i [MWh] × ef_i [g/kWh]) / total_gen [MWh]
                         = Σ(gen_i × ef_i) / Σ(gen_i)   [g/kWh]

    Returns:
        pd.Series of carbon intensity (g CO₂-eq / kWh), NaN where total
        generation is zero or unavailable.
    """
    total_emissions = pd.Series(0.0, index=df.index, dtype="float64")
    total_gen       = pd.Series(0.0, index=df.index, dtype="float64")

    for col, ef in _EMISSION_FACTORS_G_KWH.items():
        if col not in df.columns:
            continue
        gen = df[col].fillna(0.0).astype("float64")
        total_emissions += gen * ef
        total_gen       += gen

    # Guard against division by zero (e.g. maintenance windows, data gaps)
    total_gen_safe = total_gen.replace(0.0, float("nan"))
    ci = (total_emissions / total_gen_safe).astype("float32")
    ci.name = "carbon_intensity_g_kwh"

    non_nan = ci.notna().sum()
    logger.info(
        "Carbon intensity computed: %d non-NaN rows | "
        "mean=%.1f g/kWh, min=%.1f, max=%.1f",
        non_nan,
        ci.mean() if non_nan > 0 else float("nan"),
        ci.min()  if non_nan > 0 else float("nan"),
        ci.max()  if non_nan > 0 else float("nan"),
    )
    return ci


# ── Public loader ─────────────────────────────────────────────────────────────

def load_smard(
    raw_dir: Path,
    processed_dir: Path,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load all SMARD filters and return a merged hourly DataFrame.

    Includes the full generation mix (renewables + conventional), load,
    day-ahead price, cross-border flows, and derived aggregate columns:
      - total_generation_mwh        (renewables + conventional)
      - total_renewables_mwh
      - total_conventional_mwh
      - renewable_share              (fraction 0–1)
      - carbon_intensity_g_kwh       (g CO₂-eq per kWh of grid electricity)

    Returns:
        Hourly DataFrame indexed by UTC timestamp.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache = processed_dir / "smard.parquet"

    if cache.exists() and not force:
        logger.info("Loading cached SMARD parquet from %s", cache)
        return pd.read_parquet(cache)

    all_series: dict[str, pd.Series] = {}
    for filter_id, col_name in SMARD_FILTERS.items():
        s = download_smard_series(
            filter_id, raw_dir, start=start, end=end, force=force
        )
        all_series[col_name] = s

    df = pd.DataFrame(all_series)
    df.index.name = "timestamp"
    df = df.sort_index()

    # ── Derived aggregate columns ─────────────────────────────────────────
    ren_available  = [c for c in _RENEWABLE_COLS if c in df.columns]
    conv_available = [c for c in _CONVENTIONAL_COLS if c in df.columns]

    df["total_renewables_mwh"]    = df[ren_available].sum(axis=1, min_count=1)
    df["total_conventional_mwh"]  = df[conv_available].sum(axis=1, min_count=1)
    df["total_generation_mwh"]    = (
        df["total_renewables_mwh"] + df["total_conventional_mwh"]
    )

    # Renewable share — guard against division by zero
    total = df["total_generation_mwh"].replace(0, float("nan"))
    df["renewable_share"] = (df["total_renewables_mwh"] / total).astype("float32")

    # ── Carbon intensity ─────────────────────────────────────────────────
    # Computed after total_generation_mwh is available so the logger can
    # report a meaningful percentage of total generation.
    df["carbon_intensity_g_kwh"] = _compute_carbon_intensity(df)

    logger.info(
        "SMARD loaded: %d rows, %s → %s | columns: %s",
        len(df),
        df.index.min().date(),
        df.index.max().date(),
        sorted(df.columns.tolist()),
    )

    df.to_parquet(cache)
    return df
