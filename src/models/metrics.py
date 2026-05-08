"""
Evaluation metrics for energy forecasting.

All functions accept array-like inputs (numpy arrays, pandas Series/DataFrame)
and return plain Python floats or dicts — no framework dependencies.

Metrics implemented
───────────────────
  MAE    – Mean Absolute Error              [MW]        lower is better
  RMSE   – Root Mean Squared Error          [MW]        lower is better
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

ZERO_THRESHOLD = 1.0   # MW — values below this are excluded from MAPE


# ── Core metric functions ─────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
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
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, np.abs(y_true - y_pred) / denom, 0.0)
    return float(np.mean(ratio) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalised by the range of y_true."""
    y_true, y_pred = _clean(y_true, y_pred)
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

    Assumes y_true and y_pred are shaped (n_samples, H) where H is the
    forecast horizon. If 1-D, treats each element as a separate 1-step
    forecast and returns a single-row DataFrame.

    Args:
        y_true:       Actual values, shape (n_samples,) or (n_samples, H).
        y_pred:       Predicted values, same shape as y_true.
        max_horizon:  Truncate to first max_horizon steps if set.

    Returns:
        DataFrame with columns [horizon, mae, rmse], one row per step.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    H = y_true.shape[1]
    if max_horizon:
        H = min(H, max_horizon)

    rows = []
    for h in range(H):
        yt, yp = _clean(y_true[:, h], y_pred[:, h])
        rows.append({
            "horizon": h + 1,
            "mae":     float(np.mean(np.abs(yt - yp))),
            "rmse":    float(np.sqrt(np.mean((yt - yp) ** 2))),
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
        **meta,
    ) -> "MetricResult":
        m = all_metrics(y_true, y_pred)
        return cls(model=model, target=target, split=split,
                   horizon=horizon, meta=meta, **m)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.model}] {self.target} | {self.split} | H={self.horizon}h\n"
            f"  MAE={self.mae:.1f} MW  RMSE={self.rmse:.1f} MW  "
            f"MAPE={self.mape:.2f}%  R²={self.r2:.4f}"
        )


# ── Results registry ──────────────────────────────────────────────────────────

class ResultsRegistry:
    """
    Accumulates MetricResult objects and saves them to a JSON lines file.

    Usage:
        registry = ResultsRegistry(Path("results/metrics.jsonl"))
        registry.add(result)          # appends immediately to disk
        df = registry.to_dataframe()  # load all results as DataFrame
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, result: MetricResult) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
        logger.info("Logged result: %s", result)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    def leaderboard(self, target: str = "load_mw", split: str = "test") -> pd.DataFrame:
        """Return models ranked by MAE for a given target and split."""
        df = self.to_dataframe()
        subset = df[(df["target"] == target) & (df["split"] == split)]
        return subset.sort_values("mae")[
            ["model", "target", "split", "horizon", "mae", "rmse", "mape", "r2"]
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
