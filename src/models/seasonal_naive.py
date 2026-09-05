"""Seasonal-naive daily and weekly forecasting baselines."""

import numpy as np
import pandas as pd

from .base import BaseForecaster


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeat the observation from one daily or weekly seasonal cycle ago."""

    def __init__(
        self,
        target_col: str = "load_mw",
        horizon: int = 24,
        season_length: int = 24,
    ):
        if season_length not in {24, 168}:
            raise ValueError("season_length must be 24 or 168 hours.")
        if horizon > season_length:
            raise ValueError(
                "horizon cannot exceed season_length because that would use "
                "observations at or after the forecast origin."
            )
        super().__init__(target_col=target_col, horizon=horizon)
        self.season_length = season_length
        self.is_fitted = True

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return f"seasonal_naive_{self.season_length}"

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "SeasonalNaiveForecaster":
        """No-op retained for compatibility with the common model interface."""
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return an ``(N, horizon)`` forecast using only past observations."""
        series = df[self.target_col]
        forecasts = np.full((len(df), self.horizon), np.nan, dtype=np.float32)

        if isinstance(df.index, pd.DatetimeIndex):
            for step in range(self.horizon):
                source_index = df.index + pd.to_timedelta(
                    step - self.season_length, unit="h"
                )
                forecasts[:, step] = series.reindex(source_index).to_numpy(
                    dtype=np.float32
                )
            return forecasts

        values = series.to_numpy(dtype=np.float32)
        origins = np.arange(len(values))
        for step in range(self.horizon):
            source = origins + step - self.season_length
            valid = source >= 0
            forecasts[valid, step] = values[source[valid]]
        return forecasts

    def save(self, path) -> None:
        """Seasonal naive has no learned state to persist."""
        raise NotImplementedError("SeasonalNaiveForecaster has no learned state.")

    @classmethod
    def load(cls, path) -> "SeasonalNaiveForecaster":
        """Seasonal naive has no learned state to restore."""
        raise NotImplementedError("Construct SeasonalNaiveForecaster directly.")
