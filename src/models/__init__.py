# src/models/__init__.py
from importlib import import_module

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

_LAZY_MODELS = {
    "SARIMAForecaster": (".arima", "SARIMAForecaster", True),
    "XGBoostForecaster": (".xgboost_model", "XGBoostForecaster", True),
    "SeasonalNaiveForecaster": (
        ".seasonal_naive",
        "SeasonalNaiveForecaster",
        False,
    ),
    "TimesFMForecaster": (".timesfm_model", "TimesFMForecaster", False),
    "MoiraiForecaster": (".moirai_model", "MoiraiForecaster", False),
}


def __getattr__(name):
    """Load model adapters only when callers explicitly request them."""
    if name not in _LAZY_MODELS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute, optional_none = _LAZY_MODELS[name]
    try:
        value = getattr(import_module(module_name, __name__), attribute)
    except ImportError:
        if not optional_none:
            raise
        value = None
    globals()[name] = value
    return value


__all__ = [
    "BaseForecaster",
    "SARIMAForecaster",
    "XGBoostForecaster",
    "SeasonalNaiveForecaster",
    "TimesFMForecaster",
    "MoiraiForecaster",
    "MetricResult",
    "ResultsRegistry",
    "all_metrics",
    "eval_by_horizon",
    "mae", "rmse", "mape", "smape",
]