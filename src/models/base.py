"""
Base class for all forecasting models in this project.

Every model — baseline or foundation — implements this interface so that
the evaluation harness, continual learning loop, and notebooks can treat
them identically regardless of their internals.

Contract
────────
  fit(train_df, val_df)
      Train the model. val_df is available for early stopping / HPO.
      Must set self.is_fitted = True on success.

  predict(df, horizon)
      Return a 2-D numpy array of shape (len(df), horizon).
      df is the feature DataFrame at the forecast origin points.
      Each row i is a forecast issued at df.index[i], covering steps
      i+1, i+2, …, i+horizon into the future.

  save(path) / load(path)
      Serialise and restore model state. Subclasses override these.

  evaluate(df, target_col, horizon, split_name)
      Convenience wrapper: calls predict(), computes all metrics,
      returns a MetricResult. Used by the evaluation harness.

Design notes
────────────
  - predict() receives the FEATURE DataFrame, not raw targets. The model
    is responsible for selecting which columns it uses internally.
  - Multi-step forecasting strategy is up to the subclass:
      * Direct:    train one model per horizon step (XGBoost default)
      * Recursive: feed prediction back as a lag (ARIMA default)
      * Seq2Seq:   encoder-decoder (foundation models)
  - All models must be able to re-fit incrementally via fit() with new
    data (for the continual learning loop in Phase 5).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import MetricResult, all_metrics

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all energy forecasting models."""

    # Subclasses set this to a short unique string, e.g. "arima", "xgboost"
    model_name: str = "base"

    def __init__(self, target_col: str, horizon: int = 24):
        """
        Args:
            target_col: Which column in the feature DataFrame to forecast.
            horizon:    Number of steps ahead to predict (hours).
        """
        self.target_col = target_col
        self.horizon    = horizon
        self.is_fitted  = False
        self._logger    = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "BaseForecaster":
        """
        Fit the model on training data.

        Args:
            train_df: Feature DataFrame for training (includes target column).
            val_df:   Optional validation DataFrame for early stopping / HPO.

        Returns:
            self (for chaining).
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate multi-step forecasts from each row of df.

        Args:
            df: Feature DataFrame at forecast origin points.

        Returns:
            Array of shape (len(df), self.horizon) with forecasts in MW.
            If the model only produces 1-step forecasts, shape is (len(df), 1).
        """
        ...

    # ── Concrete helpers (shared by all subclasses) ───────────────────────

    def evaluate(
        self,
        df: pd.DataFrame,
        split_name: str = "test",
    ) -> MetricResult:
        """
        Run predict(), align with actuals, compute all metrics.

        For multi-step evaluation we report the *average* error across
        all horizon steps (standard practice for comparative baselines).
        Per-step breakdown is available via metrics.eval_by_horizon().

        Args:
            df:         Feature DataFrame (must contain self.target_col).
            split_name: Label for the result (e.g. "val", "test").

        Returns:
            MetricResult with all scalar metrics populated.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling evaluate().")

        preds = self.predict(df)          # (N, H)
        actuals = df[self.target_col].values  # (N,)

        # For multi-step: compare h=1 predictions against actuals
        # (the most common single-number summary for baseline tables)
        y_pred_h1 = preds[:, 0] if preds.ndim == 2 else preds

        result = MetricResult.from_arrays(
            model=self.model_name,
            target=self.target_col,
            split=split_name,
            horizon=self.horizon,
            y_true=actuals,
            y_pred=y_pred_h1,
        )
        self._logger.info(str(result))
        return result

    def save(self, path: str | Path) -> None:
        """Persist model to disk. Subclasses must override."""
        raise NotImplementedError(f"{self.model_name}.save() not implemented.")

    @classmethod
    def load(cls, path: str | Path) -> "BaseForecaster":
        """Restore model from disk. Subclasses must override."""
        raise NotImplementedError(f"{cls.__name__}.load() not implemented.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return (
            f"{self.__class__.__name__}("
            f"target='{self.target_col}', horizon={self.horizon}h, {status})"
        )
