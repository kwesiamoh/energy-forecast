"""
Evaluation metrics for energy forecasting.

All functions accept array-like inputs (numpy arrays, pandas Series/DataFrame)
and return plain Python floats or dicts — no framework dependencies.

Metrics implemented
───────────────────
  MAE    – Mean Absolute Error              [target unit] lower is better
  RMSE   – Root Mean Squared Error          [target unit] lower is better
  MAPE   – Mean Absolute Percentage Error   [%]         lower is better
           (skipped for near-zero values to avoid division by zero)
  sMAPE  – Symmetric MAPE                  [%]         lower is better
  R²     – Coefficient of determination    [−1, 1]     higher is better
  NRMSE  – RMSE normalised by range        [0, 1]      lower is better

Horizon-aware evaluation
─────────────────────────
  eval_by_horizon() computes MAE/RMSE per forecast step (h=1,2,…,H) so you
  can see how quickly accuracy degrades. Essential for comparing models whose
  error profiles differ by horizon (e.g. ARIMA degrades steeply, XGBoost
  degrades more gracefully with lag features).

Result container
─────────────────
  MetricResult dataclass — serialisable to dict/JSON for experiment tracking.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ZERO_THRESHOLD = 1.0   # target-unit values below this are excluded from MAPE

TARGET_UNITS = {
    "carbon_intensity_g_kwh": "g/kWh",
}
DEFAULT_TARGET_UNIT = "MW"


def get_target_unit(target: str) -> str:
    """Return the reporting unit for a forecast target."""
    return TARGET_UNITS.get(target, DEFAULT_TARGET_UNIT)


# ── Core metric functions ─────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = ZERO_THRESHOLD) -> float:
    """MAPE excluding near-zero actuals (avoids inf for solar at night)."""
    y_true, y_pred = _clean(y_true, y_pred)
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — bounded [0, 200%], handles zeros gracefully."""
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, np.abs(y_true - y_pred) / denom, 0.0)
    return float(np.mean(ratio) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalised by the range of y_true."""
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    r = float(y_true.max() - y_true.min())
    return rmse(y_true, y_pred) / r if r > 0 else float("nan")


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return all metrics as a flat dict."""
    return {
        "mae":   mae(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "mape":  mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2":    r2(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
    }


# ── Horizon-aware evaluation ──────────────────────────────────────────────────

def eval_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_horizon: int | None = None,
) -> pd.DataFrame:
    """
    Compute MAE and RMSE for each forecast step h = 1, 2, …, H.

    For one-dimensional actuals and a prediction matrix, column h is aligned
    as ``y_pred[:N-h, h]`` against ``y_true[h:]``. Two-dimensional actuals
    are treated as an already-aligned target matrix.

    Args:
        y_true:       Actual values, shape (n_samples,) or (n_samples, H).
        y_pred:       Predicted values, shape (n_samples,) or (n_samples, H).
        max_horizon:  Truncate to first max_horizon steps if set.

    Returns:
        DataFrame with columns [horizon, mae, rmse], one row per step.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.ndim not in (1, 2) or y_pred.ndim != 2:
        raise ValueError("y_true must be 1-D or 2-D and y_pred must be 1-D or 2-D")

    H = y_pred.shape[1]
    if y_true.ndim == 2:
        H = min(H, y_true.shape[1])
    if max_horizon:
        H = min(H, max_horizon)

    rows = []
    for h in range(H):
        if y_true.ndim == 1:
            n = min(len(y_true), len(y_pred))
            yt = y_true[h:n]
            yp = y_pred[:max(n - h, 0), h]
        else:
            n = min(len(y_true), len(y_pred))
            yt = y_true[:n, h]
            yp = y_pred[:n, h]
        rows.append({
            "horizon": h + 1,
            "mae":     mae(yt, yp),
            "rmse":    rmse(yt, yp),
        })

    return pd.DataFrame(rows)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    """Serialisable container for one model's evaluation on one split."""
    model:   str
    target:  str
    split:   str                          # "val" or "test"
    horizon: int                          # forecast horizon in hours
    mae:     float = 0.0
    rmse:    float = 0.0
    mape:    float = 0.0
    smape:   float = 0.0
    r2:      float = 0.0
    nrmse:   float = 0.0
    evaluated_step: int = 1               # one-based forecast step summarized
    sample_count: int = 0                 # finite actual/prediction pairs
    quick: bool = False                   # subset/smoke evaluation
    meta:    dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        model: str,
        target: str,
        split: str,
        horizon: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        evaluated_step: int = 1,
        quick: bool = False,
        **meta,
    ) -> "MetricResult":
        m = all_metrics(y_true, y_pred)
        clean_true, _ = _clean(y_true, y_pred)
        return cls(model=model, target=target, split=split,
                   horizon=horizon, evaluated_step=evaluated_step,
                   sample_count=int(clean_true.size), quick=quick,
                   meta=meta, **m)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        unit = get_target_unit(self.target)
        return (
            f"[{self.model}] {self.target} | {self.split} | "
            f"step=h{self.evaluated_step} | configured horizon={self.horizon}h\n"
            f"  MAE={self.mae:.1f} {unit}  RMSE={self.rmse:.1f} {unit}  "
            f"MAPE={self.mape:.2f}%  R²={self.r2:.4f}"
        )


# ── Results registry ──────────────────────────────────────────────────────────

class ResultsRegistry:
    """
    Stores MetricResult objects in a JSON lines file using deterministic
    replacement keys so rerunning the same evaluation does not add duplicates.

    Usage:
        registry = ResultsRegistry(Path("results/metrics.jsonl"))
        registry.add(result)          # inserts or replaces immediately on disk
        df = registry.to_dataframe()  # load all results as DataFrame
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, result: MetricResult) -> None:
        record = result.to_dict()
        rows = self._read_rows()
        key = self._key(record)
        rows = [row for row in rows if self._key(row) != key]
        rows.append(record)
        with open(self.path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        logger.info("Logged result: %s", result)

    def _read_rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _key(row: dict) -> tuple:
        return (
            row.get("model"), row.get("target"), row.get("split"),
            row.get("horizon"), row.get("evaluated_step", 1),
            row.get("quick", False),
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = self._read_rows()
        for row in rows:
            row.setdefault("evaluated_step", 1)
            row.setdefault("sample_count", 0)
            row.setdefault("quick", False)
        return pd.DataFrame(rows)

    def leaderboard(self, target: str = "load_mw", split: str = "test") -> pd.DataFrame:
        """Return models ranked by MAE for a given target and split."""
        df = self.to_dataframe()
        subset = df[(df["target"] == target) & (df["split"] == split)]
        return subset.sort_values("mae")[
            ["model", "target", "split", "horizon", "evaluated_step",
             "sample_count", "quick", "mae", "rmse", "mape", "r2"]
        ].reset_index(drop=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaN pairs and cast to float64."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]
