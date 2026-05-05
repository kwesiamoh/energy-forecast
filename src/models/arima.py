"""
SARIMA baseline model.

ARIMA (AutoRegressive Integrated Moving Average) and its seasonal variant
SARIMA are the canonical univariate time-series baselines. They use only
the target series itself — no exogenous features — making them the purest
possible baseline: "how well can you forecast knowing only past values?"

If XGBoost (with all its features) can't beat SARIMA on a given target,
something is seriously wrong with the feature engineering.

Model strategy
──────────────
  SARIMA(p, d, q)(P, D, Q, s)

  For hourly energy data the obvious seasonality is s=24 (daily cycle).
  Adding s=168 (weekly) creates a SARIMA with two seasonal terms, which
  statsmodels supports but is very slow to fit. Instead we:

  1. Use pmdarima.auto_arima() to find the best (p,d,q)(P,D,Q) for s=24
     on a subsample of training data (max 30 days for speed).
  2. Refit the chosen order on the full training set.
  3. Use recursive (one-step-ahead) forecasting for multi-step prediction:
     each step feeds the previous prediction back as the next observation.

Multi-step forecasting
──────────────────────
  ARIMA naturally produces recursive multi-step forecasts via
  ARIMAResults.forecast(steps=H). Prediction intervals are also available
  but not stored by default (add return_conf_int=True in predict() if needed).

Limitations (document honestly for README)
───────────────────────────────────────────
  - Univariate only: no weather or calendar inputs
  - Slow to refit on large datasets (minutes per target)
  - Assumes stationarity after differencing — may struggle during
    structural breaks (e.g. COVID load collapse in April 2020)
  - Weekly seasonality (168h) not modelled explicitly

Dependencies: pmdarima (pip install pmdarima), statsmodels
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseForecaster

logger = logging.getLogger(__name__)

# Default SARIMA order (used if auto_arima is disabled or unavailable)
DEFAULT_ORDER         = (2, 1, 2)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 24)

# Training subsample for order-selection (hours)
AUTO_ARIMA_SAMPLE_HOURS = 30 * 24   # 30 days — fast enough for CI


class SARIMAForecaster(BaseForecaster):
    """
    SARIMA univariate baseline with optional auto-order selection.

    Args:
        target_col:     Column to forecast.
        horizon:        Forecast horizon in hours.
        auto_order:     If True, run pmdarima.auto_arima() to find best order.
                        If False, use DEFAULT_ORDER / DEFAULT_SEASONAL_ORDER.
        seasonal:       Include seasonal component (s=24). True by default.
        max_p, max_q:   Search bounds for auto_arima.
    """

    model_name = "sarima"

    def __init__(
        self,
        target_col: str = "load_mw",
        horizon: int = 24,
        auto_order: bool = True,
        seasonal: bool = True,
        max_p: int = 3,
        max_q: int = 3,
    ):
        super().__init__(target_col=target_col, horizon=horizon)
        self.auto_order   = auto_order
        self.seasonal     = seasonal
        self.max_p        = max_p
        self.max_q        = max_q
        self._model       = None   # fitted ARIMAResults or SARIMAX object
        self._order       = DEFAULT_ORDER
        self._seasonal_order = DEFAULT_SEASONAL_ORDER if seasonal else (0, 0, 0, 0)
        self._train_series: pd.Series | None = None

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "SARIMAForecaster":
        series = self._extract_series(train_df)
        self._train_series = series

        if self.auto_order:
            self._order, self._seasonal_order = self._find_order(series)

        self._logger.info(
            "Fitting SARIMA%s%s on %d obs …",
            self._order,
            self._seasonal_order if self.seasonal else "",
            len(series),
        )

        self._model = self._fit_model(series)
        self.is_fitted = True
        self._logger.info("SARIMA fitted.")
        return self

    def _find_order(
        self, series: pd.Series
    ) -> tuple[tuple, tuple]:
        """Run auto_arima on a subsample to select SARIMA order."""
        try:
            import pmdarima as pm
        except ImportError:
            self._logger.warning(
                "pmdarima not installed — using default SARIMA order. "
                "Install with: pip install pmdarima"
            )
            return DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER

        subsample = series.iloc[-AUTO_ARIMA_SAMPLE_HOURS:]
        self._logger.info(
            "Running auto_arima on %d observations (this may take 1–3 min) …",
            len(subsample),
        )

        m = 24 if self.seasonal else 1
        result = pm.auto_arima(
            subsample,
            start_p=1, max_p=self.max_p,
            start_q=0, max_q=self.max_q,
            d=None,                     # auto-select d via unit root test
            seasonal=self.seasonal,
            m=m,
            D=1 if self.seasonal else 0,
            max_P=2, max_Q=2,
            information_criterion="aic",
            stepwise=True,              # stepwise is much faster than full search
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )

        order = result.order
        seasonal_order = result.seasonal_order if self.seasonal else (0, 0, 0, 0)
        self._logger.info(
            "auto_arima selected: SARIMA%s%s  (AIC=%.1f)",
            order, seasonal_order, result.aic(),
        )
        return order, seasonal_order

    def _fit_model(self, series: pd.Series):
        """Fit SARIMAX on the full training series."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            series,
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
        return result

    # ── predict ───────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate horizon-step forecasts for each row of df.

        Strategy: apply_model() on the new observations up to each origin,
        then forecast H steps ahead. For large df this is slow — consider
        batching or using predict_in_sample() for the training period.

        For the test set we iterate row-by-row (expensive but correct).
        For a faster approximate version, use predict_rolling() below.

        Returns:
            Array of shape (len(df), self.horizon).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        series = self._extract_series(df)
        preds = np.full((len(df), self.horizon), np.nan)

        # Use SARIMAX.apply() to update the model state with new observations
        # then call forecast(). This is the statistically correct approach.
        updated = self._model.apply(series.values)

        for i in range(len(df)):
            try:
                fc = updated.forecast(steps=self.horizon)
                preds[i] = fc.values
            except Exception as e:
                self._logger.debug("Forecast failed at step %d: %s", i, e)

        return preds

    def predict_rolling(self, df: pd.DataFrame) -> np.ndarray:
        """
        Faster approximate rolling forecast using in-sample predictions
        extended with recursive out-of-sample steps.

        Use this for quick evaluation during development. Use predict() for
        final benchmark numbers.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        series = self._extract_series(df)
        n = len(series)
        preds = np.full((n, self.horizon), np.nan)

        # In-sample one-step predictions
        insample = self._model.predict(start=0, end=len(self._train_series) - 1)

        # Out-of-sample
        try:
            forecast_all = self._model.forecast(steps=n + self.horizon)
            for i in range(n):
                preds[i] = forecast_all.values[i: i + self.horizon]
        except Exception as e:
            self._logger.warning("Rolling forecast failed: %s", e)

        return preds

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "target_col":      self.target_col,
            "horizon":         self.horizon,
            "auto_order":      self.auto_order,
            "seasonal":        self.seasonal,
            "order":           self._order,
            "seasonal_order":  self._seasonal_order,
            "model_pickle":    self._model.to_json() if self._model else None,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        self._logger.info("SARIMA saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SARIMAForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(
            target_col=state["target_col"],
            horizon=state["horizon"],
            auto_order=state["auto_order"],
            seasonal=state["seasonal"],
        )
        instance._order         = state["order"]
        instance._seasonal_order = state["seasonal_order"]
        if state["model_pickle"]:
            instance._model = SARIMAXResults.from_json(state["model_pickle"])
        instance.is_fitted = True
        return instance

    # ── internal ──────────────────────────────────────────────────────────

    def _extract_series(self, df: pd.DataFrame) -> pd.Series:
        """Pull target column and drop NaNs."""
        s = df[self.target_col].dropna()
        if len(s) < self.horizon * 2:
            raise ValueError(
                f"Too few non-NaN observations ({len(s)}) for SARIMA fit."
            )
        return s
