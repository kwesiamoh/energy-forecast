"""Lazy zero-shot adapter for the official TimesFM 2.5 PyTorch model."""

import time

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .device import resolve_device, resolve_dtype
from .foundation_utils import iter_context_batches


DEFAULT_MODEL = "google/timesfm-2.5-200m-pytorch"


class TimesFMForecaster(BaseForecaster):
    """Target-only TimesFM 2.5 forecasts with a shared ``(N, H)`` contract."""

    model_name = "timesfm_2_5"

    def __init__(
        self,
        target_col: str = "load_mw",
        horizon: int = 24,
        model_id: str = DEFAULT_MODEL,
        context_length: int = 168,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 16,
    ):
        super().__init__(target_col=target_col, horizon=horizon)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.model_id = model_id
        self.context_length = context_length
        self.batch_size = batch_size
        self._device_spec = resolve_device(device)
        self.device = self._device_spec.name
        self.dtype, _ = resolve_dtype(
            dtype, self._device_spec, reduced_precision=False
        )
        self._model = None
        self.last_quantile_forecasts: dict[float, np.ndarray] = {}
        self.last_runtime_seconds: float | None = None
        self.is_fitted = True

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "TimesFMForecaster":
        """No-op: this adapter intentionally supports zero-shot inference only."""
        self.is_fitted = True
        return self

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "TimesFM 2.5 is optional. Install 'timesfm[torch]==2.0.2'."
            ) from exc

        try:
            model_class = timesfm.TimesFM_2p5_200M_torch
            forecast_config = timesfm.ForecastConfig
        except AttributeError as exc:
            raise ImportError(
                "The installed TimesFM package does not expose the 2.5 PyTorch "
                "API. Install 'timesfm[torch]==2.0.2'."
            ) from exc

        self._model = model_class.from_pretrained(
            self.model_id, torch_compile=False
        )
        self._model.model.to(self.device)
        self._model.model.device = self._torch_device()
        self._model.model.device_count = 1
        self._model.compile(
            forecast_config(
                max_context=self.context_length,
                max_horizon=self.horizon,
                normalize_inputs=True,
                per_core_batch_size=self.batch_size,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._logger.info(
            "Model: TimesFM 2.5 | Device: %s | GPU: %s | dtype: %s | Batch size: %d",
            self.device,
            self._device_spec.gpu_name or "CPU",
            self.dtype,
            self.batch_size,
        )

    def _torch_device(self):
        import torch

        return torch.device(self.device)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Forecast each valid origin from the preceding target-only context."""
        import torch

        self._load_model()
        values = df[self.target_col].to_numpy(dtype=np.float32)
        point = np.full((len(values), self.horizon), np.nan, dtype=np.float32)
        quantiles = {
            q: np.full_like(point, np.nan) for q in (0.1, 0.5, 0.9)
        }
        started = time.perf_counter()
        with torch.inference_mode():
            for start, contexts in iter_context_batches(
                values, self.context_length, self.batch_size
            ):
                batch_point, batch_quantiles = self._model.forecast(
                    horizon=self.horizon,
                    inputs=[row for row in contexts],
                )
                stop = start + len(contexts)
                point[start:stop] = np.asarray(batch_point, dtype=np.float32)
                batch_q = np.asarray(batch_quantiles, dtype=np.float32)
                quantiles[0.1][start:stop] = batch_q[:, :, 1]
                quantiles[0.5][start:stop] = batch_q[:, :, 5]
                quantiles[0.9][start:stop] = batch_q[:, :, 9]

        self.last_runtime_seconds = time.perf_counter() - started
        self.last_quantile_forecasts = quantiles
        self._logger.info(
            "TimesFM 2.5 inference complete: target=%s, origins=%d, elapsed=%.2fs",
            self.target_col,
            max(0, len(values) - self.context_length),
            self.last_runtime_seconds,
        )
        return point

    def predict_quantiles(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        """Run inference and return the model's genuine 10/50/90% quantiles."""
        self.predict(df)
        return self.last_quantile_forecasts
