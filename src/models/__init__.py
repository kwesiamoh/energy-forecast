# src/models/__init__.py
# Core (no heavy deps)
from .base import BaseForecaster
from .metrics import (
    MetricResult,
    ResultsRegistry,
    all_metrics,
    eval_by_horizon,
    mae,
    mape,
    rmse,
    smape,
)

# Optional: SARIMA (requires statsmodels + pmdarima)
try:
    from .arima import SARIMAForecaster
except ImportError:
    SARIMAForecaster = None  # type: ignore

# Optional: XGBoost (requires xgboost)
try:
    from .xgboost_model import XGBoostForecaster
except ImportError:
    XGBoostForecaster = None  # type: ignore

__all__ = [
    "BaseForecaster",
    "SARIMAForecaster",
    "XGBoostForecaster",
    "MetricResult",
    "ResultsRegistry",
    "all_metrics",
    "eval_by_horizon",
    "mae", "rmse", "mape", "smape",
]
