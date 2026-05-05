"""
XGBoost multi-output direct forecasting model.

XGBoost is the strongest classical baseline for energy forecasting. Unlike
ARIMA it consumes ALL features — calendar, weather, lags — which typically
gives it a 20–40% MAE advantage over univariate methods.

Forecasting strategy: DIRECT multi-output
──────────────────────────────────────────
  We train H independent XGBoost regressors, one per horizon step h=1…H.
  This is the "direct" strategy:

    model_h(X_t) → ŷ_{t+h}   for h = 1, 2, …, H

  Advantages over recursive:
  - No error accumulation (each model is independently optimal for its step)
  - Parallelisable (train all H models simultaneously)
  - Feature set can differ per step (though we use the same here)

  Disadvantage: H × model storage (mitigated by XGBoost's small tree sizes).

Feature handling
────────────────
  - Accepts any numeric columns from the Phase 2 feature DataFrame.
  - Target-column lags (load_mw_lag24 etc.) are the most important features.
  - NaN rows are dropped before fitting (lag warmup period).
  - At prediction time, NaN features cause XGBoost to use its built-in
    missing-value handling (follows the optimal split direction learned
    during training).

Hyperparameter tuning
─────────────────────
  Default hyperparams are competitive out-of-the-box for energy data.
  Run XGBoostForecaster.tune() to run Optuna HPO on the validation set.
  Tuned params are stored in self.params and reused on retrain.

Dependencies: xgboost, optuna (optional, for HPO), joblib
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseForecaster

logger = logging.getLogger(__name__)

# Competitive defaults for hourly energy forecasting (no tuning required)
DEFAULT_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,     # L1 — helps with sparse solar features
    "reg_lambda":       1.0,     # L2
    "tree_method":      "hist",  # fast histogram algorithm
    "device":           "cpu",   # change to "cuda" if GPU available
    "random_state":     42,
    "verbosity":        0,
}


class XGBoostForecaster(BaseForecaster):
    """
    Direct multi-step XGBoost forecaster.

    Args:
        target_col:    Column to forecast.
        horizon:       Forecast horizon in hours (one model per step).
        feature_cols:  Explicit list of feature columns. If None, all numeric
                       columns except target and SMARD overlays are used.
        params:        XGBoost hyperparameters (merged with DEFAULT_PARAMS).
        early_stopping_rounds:
                       Stop training if val loss doesn't improve for N rounds.
                       Set to None to disable (faster, slightly lower quality).
    """

    model_name = "xgboost"

    def __init__(
        self,
        target_col: str = "load_mw",
        horizon: int = 24,
        feature_cols: list[str] | None = None,
        params: dict | None = None,
        early_stopping_rounds: int | None = 30,
    ):
        super().__init__(target_col=target_col, horizon=horizon)
        self.feature_cols           = feature_cols
        self.params                 = {**DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds  = early_stopping_rounds
        self._models: list[xgb.XGBRegressor] = []  # one per horizon step
        self._feature_cols_fitted: list[str] = []

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "XGBoostForecaster":
        """
        Train H XGBoost regressors (one per horizon step).

        For step h, the label is the target value h rows ahead:
            y_h[i] = target[i + h]

        This means the training set shrinks by h rows at the end for each
        model — a small cost for correct direct forecasting.
        """
        feat_cols = self._resolve_features(train_df)
        self._feature_cols_fitted = feat_cols
        self._models = []

        X_train = train_df[feat_cols]
        y_train_base = train_df[self.target_col]

        X_val = val_df[feat_cols] if val_df is not None else None
        y_val_base = val_df[self.target_col] if val_df is not None else None

        self._logger.info(
            "Fitting %d XGBoost models for '%s' (H=1…%d) on %d rows, %d features …",
            self.horizon, self.target_col, self.horizon, len(X_train), len(feat_cols),
        )

        for h in range(1, self.horizon + 1):
            # Shift target h steps into the future relative to features
            y_h = y_train_base.shift(-h)

            # Align: drop rows where y or any feature is NaN
            mask_train = X_train.notna().all(axis=1) & y_h.notna()
            X_h = X_train[mask_train].values
            y_h = y_h[mask_train].values

            fit_kwargs: dict = {}
            if X_val is not None and y_val_base is not None and self.early_stopping_rounds:
                y_val_h = y_val_base.shift(-h)
                mask_val = X_val.notna().all(axis=1) & y_val_h.notna()
                X_val_h = X_val[mask_val].values
                y_val_h = y_val_base.shift(-h)[mask_val].values
                fit_kwargs["eval_set"] = [(X_val_h, y_val_h)]
                fit_kwargs["verbose"] = False

            model = xgb.XGBRegressor(
                **self.params,
                early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            )
            model.fit(X_h, y_h, **fit_kwargs)
            self._models.append(model)

            if h % 6 == 0 or h == self.horizon:
                self._logger.info("  … h=%d/%d fitted", h, self.horizon)

        self.is_fitted = True
        self._logger.info(
            "XGBoost fitting complete. Feature importances available via .feature_importance()"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate H-step forecasts for each row of df.

        Returns:
            Array of shape (len(df), self.horizon).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        X = df[self._feature_cols_fitted].values
        preds = np.full((len(df), self.horizon), np.nan, dtype=np.float32)

        for h, model in enumerate(self._models):
            preds[:, h] = model.predict(X)

        return preds

    # ── feature importance ────────────────────────────────────────────────

    def feature_importance(
        self,
        horizon_step: int = 1,
        importance_type: str = "gain",
        top_n: int = 30,
    ) -> pd.Series:
        """
        Return feature importances for a given horizon step.

        Args:
            horizon_step:    1-indexed step (1 = next hour, 24 = 24h ahead).
            importance_type: "gain", "weight", or "cover".
            top_n:           Return only the top N features.

        Returns:
            pd.Series indexed by feature name, sorted descending.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        model = self._models[horizon_step - 1]
        scores = model.get_booster().get_score(importance_type=importance_type)
        series = pd.Series(scores, index=list(scores.keys())).sort_values(ascending=False)
        return series.head(top_n)

    # ── HPO ───────────────────────────────────────────────────────────────

    def tune(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        n_trials: int = 50,
        horizon_step: int = 1,
    ) -> dict:
        """
        Run Optuna HPO to find better hyperparameters.

        Optimises only for horizon_step=1 (good proxy for all steps).
        Updates self.params with the best found parameters.

        Args:
            train_df:    Training DataFrame.
            val_df:      Validation DataFrame (objective metric evaluated here).
            n_trials:    Number of Optuna trials.
            horizon_step: Horizon step to optimise for.

        Returns:
            Best hyperparameter dict.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Install optuna: pip install optuna")

        feat_cols = self._resolve_features(train_df)
        X_train = train_df[feat_cols]
        y_train = train_df[self.target_col].shift(-horizon_step)
        mask = X_train.notna().all(axis=1) & y_train.notna()
        X_tr, y_tr = X_train[mask].values, y_train[mask].values

        X_val = val_df[feat_cols]
        y_val = val_df[self.target_col].shift(-horizon_step)
        mask_v = X_val.notna().all(axis=1) & y_val.notna()
        X_v, y_v = X_val[mask_v].values, y_val[mask_v].values

        def objective(trial):
            p = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "tree_method":      "hist",
                "random_state":     42,
                "verbosity":        0,
            }
            model = xgb.XGBRegressor(**p)
            model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
            preds = model.predict(X_v)
            return float(np.mean(np.abs(y_v - preds)))  # MAE

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        self.params = {**DEFAULT_PARAMS, **best}
        self._logger.info("Optuna HPO complete. Best MAE=%.2f. Params: %s", study.best_value, best)
        return self.params

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "target_col":           self.target_col,
            "horizon":              self.horizon,
            "params":               self.params,
            "feature_cols":         self._feature_cols_fitted,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        # Save state dict
        with open(path.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump(state, f)
        # Save each XGB model separately (supports JSON serialisation)
        for h, model in enumerate(self._models):
            model.save_model(str(path.with_suffix(f".h{h+1:03d}.ubj")))
        self._logger.info("XGBoost saved → %s (.meta.pkl + %d model files)", path, len(self._models))

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostForecaster":
        path = Path(path)
        with open(path.with_suffix(".meta.pkl"), "rb") as f:
            state = pickle.load(f)

        instance = cls(
            target_col=state["target_col"],
            horizon=state["horizon"],
            params=state["params"],
            early_stopping_rounds=state["early_stopping_rounds"],
        )
        instance._feature_cols_fitted = state["feature_cols"]
        instance._models = []
        for h in range(instance.horizon):
            model = xgb.XGBRegressor()
            model.load_model(str(path.with_suffix(f".h{h+1:03d}.ubj")))
            instance._models.append(model)
        instance.is_fitted = True
        return instance

    # ── internal ──────────────────────────────────────────────────────────

    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        """Return feature columns, excluding target and non-numeric cols."""
        if self.feature_cols is not None:
            return [c for c in self.feature_cols if c in df.columns]

        exclude = {self.target_col}
        exclude.update(c for c in df.columns if "_smard" in c)
        exclude.update(c for c in df.columns if c.startswith((
            "berlin_", "frankfurt_", "munich_", "hamburg_", "stuttgart_"
        )))

        return [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
        ]
