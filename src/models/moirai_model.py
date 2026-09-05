"""Lazy zero-shot adapter for Moirai 2.0 Small through Uni2TS/GluonTS."""

import time

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .device import resolve_device, resolve_dtype
from .foundation_utils import iter_context_batches


DEFAULT_MODEL = "Salesforce/moirai-2.0-R-small"


class MoiraiForecaster(BaseForecaster):
    """Target-only Moirai 2.0 Small forecasts using its median leaderboard output."""

    model_name = "moirai_2_0_small"

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
        self._predictor = None
        self.last_quantile_forecasts: dict[float, np.ndarray] = {}
        self.last_runtime_seconds: float | None = None
        self.is_fitted = True

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "MoiraiForecaster":
        """No-op: this adapter intentionally supports zero-shot inference only."""
        self.is_fitted = True
        return self

    def _load_predictor(self) -> None:
        if self._predictor is not None:
            return
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError as exc:
            raise ImportError(
                "Moirai 2.0 is optional. Install 'uni2ts==2.0.0'."
            ) from exc

        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self.model_id),
            prediction_length=self.horizon,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = model.create_predictor(
            batch_size=self.batch_size,
            device=self.device,
        )
        self._logger.info(
            "Model: Moirai 2.0 Small | Device: %s | GPU: %s | dtype: %s | Batch size: %d",
            self.device,
            self._device_spec.gpu_name or "CPU",
            self.dtype,
            self.batch_size,
        )

    def predict_contexts(self, contexts: np.ndarray) -> np.ndarray:
        """Forecast directly from prebuilt ``(N, context_length)`` contexts."""
        import torch

        try:
            from gluonts.dataset.common import ListDataset
        except ImportError as exc:
            raise ImportError(
                "Moirai 2.0 requires the GluonTS dependency installed by "
                "'uni2ts==2.0.0'."
            ) from exc

        self._load_predictor()
        contexts = np.asarray(contexts, dtype=np.float32)
        if contexts.ndim != 2 or contexts.shape[1] != self.context_length:
            raise ValueError(
                "Moirai contexts must have shape "
                f"(N, {self.context_length}); got {contexts.shape}."
            )

        point = np.full(
            (len(contexts), self.horizon), np.nan, dtype=np.float32
        )
        quantiles = {
            q: np.full_like(point, np.nan) for q in (0.1, 0.5, 0.9)
        }
        started = time.perf_counter()
        base_start = pd.Period("2000-01-01 00:00", freq="h")

        with torch.inference_mode():
            for start in range(0, len(contexts), self.batch_size):
                batch = contexts[start:start + self.batch_size]
                dataset = ListDataset(
                    [
                        {"start": base_start, "target": context}
                        for context in batch
                    ],
                    freq="h",
                )
                forecasts = list(self._predictor.predict(dataset))
                for offset, forecast in enumerate(forecasts):
                    row = start + offset
                    for q in quantiles:
                        forecast_q = np.asarray(
                            forecast.quantile(q), dtype=np.float32
                        ).squeeze()
                        quantiles[q][row] = forecast_q.reshape(-1)[
                            -self.horizon:
                        ]
                stop = start + len(forecasts)
                point[start:stop] = quantiles[0.5][start:stop]

        self.last_runtime_seconds = time.perf_counter() - started
        self.last_quantile_forecasts = quantiles
        self._logger.info(
            "Moirai 2.0 Small inference complete: target=%s, contexts=%d, elapsed=%.2fs",
            self.target_col,
            len(contexts),
            self.last_runtime_seconds,
        )
        return point
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Forecast each valid origin from the preceding target-only context."""
        import torch

        try:
            from gluonts.dataset.common import ListDataset
        except ImportError as exc:
            raise ImportError(
                "Moirai 2.0 requires the GluonTS dependency installed by "
                "'uni2ts==2.0.0'."
            ) from exc

        self._load_predictor()
        values = df[self.target_col].to_numpy(dtype=np.float32)
        point = np.full((len(values), self.horizon), np.nan, dtype=np.float32)
        quantiles = {
            q: np.full_like(point, np.nan) for q in (0.1, 0.5, 0.9)
        }
        started = time.perf_counter()
        base_start = pd.Period("2000-01-01 00:00", freq="h")

        with torch.inference_mode():
            for start, contexts in iter_context_batches(
                values, self.context_length, self.batch_size
            ):
                dataset = ListDataset(
                    [
                        {"start": base_start, "target": context}
                        for context in contexts
                    ],
                    freq="h",
                )
                forecasts = list(self._predictor.predict(dataset))
                stop = start + len(forecasts)
                for offset, forecast in enumerate(forecasts):
                    row = start + offset
                    for q in quantiles:
                        forecast_q = np.asarray(
                            forecast.quantile(q), dtype=np.float32
                        ).squeeze()
                        quantiles[q][row] = forecast_q.reshape(-1)[-self.horizon:]
                point[start:stop] = quantiles[0.5][start:stop]

        self.last_runtime_seconds = time.perf_counter() - started
        self.last_quantile_forecasts = quantiles
        self._logger.info(
            "Moirai 2.0 Small inference complete: target=%s, origins=%d, elapsed=%.2fs",
            self.target_col,
            max(0, len(values) - self.context_length),
            self.last_runtime_seconds,
        )
        return point

    def predict_quantiles(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        """Run inference and return the model's genuine 10/50/90% quantiles."""
        self.predict(df)
        return self.last_quantile_forecasts
