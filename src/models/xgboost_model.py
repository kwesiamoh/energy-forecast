"""
XGBoost recursive one-step forecaster.

XGBoost is the strongest classical baseline for energy forecasting. Unlike
ARIMA it consumes ALL features — calendar, weather, lags — which typically
gives it a 20–40% MAE advantage over univariate methods.

Forecasting strategy: RECURSIVE (single model, iterative rollout)
──────────────────────────────────────────────────────────────────
  We train one XGBoost regressor that predicts the target at t from the
  leakage-safe feature state at t:

    model(X_t) → ŷ_{t+1}

  To produce an H-step forecast, we roll the model forward, updating the
  relevant lag and rolling-window features at each step so that the next
  call to model() sees the freshly predicted value as part of its history:

    for h = 1 … H:
        ŷ_{t+h} = model(X_{t+h})
        update lag / rolling features        # inject ŷ into feature state
        X_{t+h} ← updated features          # feed forward

  Why recursive instead of direct?
  ─────────────────────────────────────────────────────────
  The direct strategy trains H independent models (one per horizon step).
  For h > 6 the training labels are so far into the future that the models
  degrade to the mean, producing a characteristic flatline.  A single h=1
  model is trained on the densest possible signal and stays sharp across
  all steps.

  Trade-off: recursive accumulates prediction errors over the rollout.
  For 24-hour energy horizons this is generally a smaller problem than the
  mean-regression artefact it replaces.

Feature handling
────────────────
  - Accepts any numeric columns from the Phase 2 feature DataFrame.
  - Target-column lags (load_mw_lag1, load_mw_lag24, etc.) and rolling
    windows (load_mw_roll24_mean, etc.) are the most important features.
  - NaN rows are dropped before fitting (lag warmup period).
  - At each recursive step only the recognised lag/rolling features for the
    target column are mutated; all other features (weather, calendar, …) are
    left untouched.

Quantile forecasts
──────────────────
  Quantile estimation via recursive point-forecasting requires residual
  bootstrapping (sample from the h=1 residual distribution and propagate
  through the rollout).  This is non-trivial and is not implemented here.
  predict_quantiles() raises NotImplementedError with guidance.

Hyperparameter tuning
─────────────────────
  Default hyperparams are competitive out-of-the-box for energy data.
  Run XGBoostForecaster.tune() to run Optuna HPO on the validation set.
  Tuned params are stored in self.params and reused on retrain.

Dependencies: xgboost, optuna (optional, for HPO), joblib
"""

import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseForecaster

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

# ---------------------------------------------------------------------------
# Regex patterns for lag and rolling features we will update during rollout.
# These patterns match the naming convention produced by Phase 2 engineering:
#
#   {target}_lag{N}              e.g. load_mw_lag1, load_mw_lag24
#   {target}_roll{N}_{stat}      e.g. load_mw_roll24_mean, load_mw_roll48_std
#   {target}_diff{N}             e.g. load_mw_diff1
#
# Only features matching these templates for the *target column* are mutated;
# all other features remain frozen at their original values.
# ---------------------------------------------------------------------------
_LAG_RE    = re.compile(r"^(.+)_lag(\d+)$")
_ROLL_RE   = re.compile(r"^(.+)_roll(\d+)_(\w+)$")
_DIFF_RE   = re.compile(r"^(.+)_diff(\d+)$")


class XGBoostForecaster(BaseForecaster):
    """
    Recursive single-step XGBoost forecaster.

    A single XGBoost model predicts the target at the current forecast row.
    Later horizons are produced by rolling it forward and injecting its output
    into the lag/rolling features at each step.

    Args:
        target_col:    Column to forecast.
        horizon:       Forecast horizon in hours.
        feature_cols:  Explicit list of feature columns. If None, all numeric
                       columns except target and raw SMARD overlays are used.
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
        feature_cols: Optional[list] = None,
        params: Optional[dict] = None,
        early_stopping_rounds: Optional[int] = 30,
    ):
        super().__init__(target_col=target_col, horizon=horizon)
        self.feature_cols            = feature_cols
        self.params                  = {**DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds   = early_stopping_rounds
        self._model: Optional[xgb.XGBRegressor] = None   # single h=1 model
        self._feature_cols_fitted: list = []

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "XGBoostForecaster":
        """
        Train one XGBoost regressor to predict the target at each feature row.

        Label:  y[i] = target[i]
        Features: all resolved feature columns at time i.

        The single model is stored in self._model.
        """
        feat_cols = self._resolve_features(train_df)
        self._feature_cols_fitted = feat_cols

        X_train = train_df[feat_cols]
        y_train = train_df[self.target_col]

        # Drop rows where any feature or the label is NaN (lag warm-up rows)
        mask_train = X_train.notna().all(axis=1) & y_train.notna()
        X_tr = X_train[mask_train].values
        y_tr = y_train[mask_train].values

        self._logger.info(
            "Fitting single XGBoost h=1 model for '%s' on %d rows, %d features …",
            self.target_col, len(X_tr), len(feat_cols),
        )

        fit_kwargs: dict = {}
        if val_df is not None and self.early_stopping_rounds:
            X_val = val_df[feat_cols]
            y_val = val_df[self.target_col]
            mask_val = X_val.notna().all(axis=1) & y_val.notna()
            X_v = X_val[mask_val].values
            y_v = y_val[mask_val].values
            fit_kwargs["eval_set"] = [(X_v, y_v)]
            fit_kwargs["verbose"] = False

        self._model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds if val_df is not None else None,
        )
        self._model.fit(X_tr, y_tr, **fit_kwargs)

        self.is_fitted = True
        self._logger.info(
            "XGBoost h=1 model fitted. Feature importances available via .feature_importance()"
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate H-step forecasts for each row of df via recursive rollout.

        Algorithm
        ─────────
        For each row i in df we maintain a *feature state* vector.  At step h:

          1. Predict ŷ_{t+h} = model(state)
          2. Store ŷ_{t+h} in the output array.
          3. Update the lag / rolling / diff features in *state* so that the
             model will see ŷ_{t+h} as recent history on the next step.

        The update step is performed in-place on a plain NumPy array (indexed
        by pre-computed column positions) to avoid any pandas fragmentation
        or SettingWithCopyWarning issues inside the loop.

        Returns:
            Array of shape (len(df), self.horizon) — one row per input row,
            one column per horizon step.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

        feat_cols = self._feature_cols_fitted
        col_index = {col: i for i, col in enumerate(feat_cols)}

        # Pre-compute which (column_index, lag_offset) pairs to update for
        # lag features, rolling features, and diff features of the target.
        lag_updates  = self._build_lag_update_plan(feat_cols, col_index)
        roll_updates = self._build_roll_update_plan(feat_cols, col_index)
        diff_updates = self._build_diff_update_plan(feat_cols, col_index)
        calendar_indices = self._calendar_feature_indices(feat_cols)

        # Work on a float32 copy — one row per sample, shape (N, F)
        # Using a separate array per sample avoids cross-row contamination.
        X_init = df[feat_cols].values.astype(np.float32)          # (N, F)

        N = len(df)
        H = self.horizon
        preds = np.full((N, H), np.nan, dtype=np.float32)

        for i in range(N):
            # Independent feature state for sample i; we will mutate this
            # in-place across the H rollout steps.
            state = X_init[i].copy()                               # (F,)

            # history_window: a rolling buffer of the last max_lag actual /
            # predicted target values so we can recompute rolling stats.
            # Populated from the lag features found in the initial state.
            history = self._extract_history_from_state(
                state, lag_updates, roll_updates, diff_updates
            )

            for h in range(H):
                # Calendar fields are deterministic and known in advance.
                # Unknown future weather and other exogenous values persist.
                future_row = i + h
                if h > 0 and future_row < N:
                    state[calendar_indices] = X_init[future_row, calendar_indices]

                # ── Step A: predict this horizon timestamp ─────────────────
                y_hat = float(self._model.predict(state.reshape(1, -1))[0])

                # ── Step B: store prediction ───────────────────────────────
                preds[i, h] = y_hat

                # ── Step C: update feature state ───────────────────────────
                self._update_state(
                    state, y_hat, history,
                    lag_updates, roll_updates, diff_updates,
                )

        return preds

    # ── feature importance ────────────────────────────────────────────────

    def feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 30,
    ) -> pd.Series:
        """
        Return feature importances for the single h=1 model.

        Args:
            importance_type: "gain", "weight", or "cover".
            top_n:           Return only the top N features.

        Returns:
            pd.Series indexed by feature name, sorted descending.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")
        scores = self._model.get_booster().get_score(importance_type=importance_type)
        series = pd.Series(scores).sort_values(ascending=False)
        return series.head(top_n)

    # ── quantile forecasts ────────────────────────────────────────────────

    def predict_quantiles(self, df: pd.DataFrame, quantiles: list) -> np.ndarray:
        """
        Not implemented for the recursive strategy.

        Quantile estimation via recursive rollout requires propagating
        uncertainty through each step — typically done via residual
        bootstrapping:

          1. Fit the h=1 model and collect its in-sample residuals.
          2. At inference time, for each bootstrap draw b:
               - Sample residuals ε_1 … ε_H (with replacement).
               - Roll out: ŷ_{t+h}^b = model(state) + ε_h, update state.
          3. Compute empirical quantiles across B draws.

        This is deferred to a future implementation.  Use predict() for
        point forecasts in the meantime.
        """
        raise NotImplementedError(
            "predict_quantiles() is not yet supported for the recursive "
            "XGBoost strategy.  Quantile intervals require residual "
            "bootstrapping across the rollout horizon — see the docstring "
            "for the recommended approach."
        )

    # ── HPO ───────────────────────────────────────────────────────────────

    def tune(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        n_trials: int = 50,
    ) -> dict:
        """
        Run Optuna HPO to find better hyperparameters for the h=1 model.

        Updates self.params with the best found parameters.

        Args:
            train_df:  Training DataFrame.
            val_df:    Validation DataFrame (objective metric evaluated here).
            n_trials:  Number of Optuna trials.

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
        y_train = train_df[self.target_col]
        mask = X_train.notna().all(axis=1) & y_train.notna()
        X_tr, y_tr = X_train[mask].values, y_train[mask].values

        X_val = val_df[feat_cols]
        y_val = val_df[self.target_col]
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
        self._logger.info(
            "Optuna HPO complete. Best MAE=%.2f. Params: %s", study.best_value, best
        )
        return self.params

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "target_col":            self.target_col,
            "horizon":               self.horizon,
            "params":                self.params,
            "feature_cols":          self._feature_cols_fitted,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        with open(path.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump(state, f)
        # Single model — save as universal binary JSON
        self._model.save_model(str(path.with_suffix(".model.ubj")))
        self._logger.info("XGBoost saved → %s (.meta.pkl + .model.ubj)", path)

    @classmethod
    def load(cls, path) -> "XGBoostForecaster":
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
        instance._model = xgb.XGBRegressor()
        instance._model.load_model(str(path.with_suffix(".model.ubj")))
        instance.is_fitted = True
        return instance

    # ── internal: feature resolution ──────────────────────────────────────

    def _resolve_features(self, df: pd.DataFrame) -> list:
        """Return the explicit or canonical pipeline feature columns."""
        if self.feature_cols is not None:
            return [c for c in self.feature_cols if c in df.columns]

        from src.features.pipeline import get_feature_cols

        return get_feature_cols(df)

    # ── internal: recursive rollout helpers ───────────────────────────────

    def _build_lag_update_plan(
        self,
        feat_cols: list,
        col_index: dict,
    ) -> list:
        """
        Return a sorted list of (col_idx, lag_int) for all lag features of
        the target column, sorted by lag ascending (lag1 first).

        Example entry: (3, 1) means feat_cols[3] is '{target}_lag1'.
        """
        plan = []
        for col in feat_cols:
            m = _LAG_RE.match(col)
            if m and m.group(1) == self.target_col:
                lag = int(m.group(2))
                plan.append((col_index[col], lag))
        # Ascending lag order so we can shift them correctly
        plan.sort(key=lambda x: x[1])
        return plan

    def _build_roll_update_plan(
        self,
        feat_cols: list,
        col_index: dict,
    ) -> list:
        """
        Return a list of (col_idx, window_int, stat_str) for all rolling
        features of the target column.

        We need the window size so we know how many history values to average.
        Supported stats: mean, std, min, max, median.
        """
        plan = []
        for col in feat_cols:
            m = _ROLL_RE.match(col)
            if m and m.group(1) == self.target_col:
                window = int(m.group(2))
                stat   = m.group(3)
                plan.append((col_index[col], window, stat))
        return plan

    def _build_diff_update_plan(
        self,
        feat_cols: list,
        col_index: dict,
    ) -> list:
        """
        Return a list of (col_idx, diff_order) for all diff features of the
        target column.

        Example: '{target}_diff1' → difference between current and lag-1.
        """
        plan = []
        for col in feat_cols:
            m = _DIFF_RE.match(col)
            if m and m.group(1) == self.target_col:
                order = int(m.group(2))
                plan.append((col_index[col], order))
        return plan

    def _extract_history_from_state(
        self,
        state: np.ndarray,
        lag_updates: list,
        roll_updates: list,
        diff_updates: list,
    ) -> list:
        """
        Reconstruct an ordered history buffer from the initial lag features.

        The buffer is a Python list where index 0 is the *most recent* known
        value (lag1), index 1 is lag2, etc.  We fill as many slots as the
        maximum lag / window we need to update.

        Values not covered by any lag feature are left as NaN.
        """
        if not lag_updates and not roll_updates:
            return []

        max_needed = 0
        if lag_updates:
            max_needed = max(max_needed, lag_updates[-1][1])   # already sorted
        if roll_updates:
            max_needed = max(max_needed, max(w for _, w, _ in roll_updates))
        if diff_updates:
            max_needed = max(max_needed, max(o for _, o in diff_updates))

        history = [np.nan] * max_needed
        for col_idx, lag in lag_updates:
            if lag <= max_needed:
                history[lag - 1] = float(state[col_idx])
        return history

    @staticmethod
    def _calendar_feature_indices(feat_cols: list[str]) -> np.ndarray:
        """Return positions of deterministic calendar features."""
        calendar_cols = {
            "hour", "dow", "month", "quarter", "week_of_year",
            "day_of_year", "year", "hour_sin", "hour_cos", "dow_sin",
            "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "is_weekend", "is_night", "is_peak_morning", "is_peak_evening",
            "is_holiday", "is_holiday_eve", "season",
        }
        return np.asarray(
            [i for i, col in enumerate(feat_cols) if col in calendar_cols],
            dtype=np.intp,
        )

    def _update_state(
        self,
        state: np.ndarray,
        y_hat: float,
        history: list,
        lag_updates: list,
        roll_updates: list,
        diff_updates: list,
    ) -> None:
        """
        Mutate *state* in-place to reflect that y_hat is the newest value.

        History is also updated in-place: y_hat is prepended (shift right),
        so history[0] always holds the most recent value.

        All writes go directly to the NumPy array via integer indices —
        no pandas involved, so no fragmentation or copy warnings.
        """
        # ── Diff features (need previous value before we shift history) ──
        for col_idx, order in diff_updates:
            if order <= len(history):
                prev = history[order - 1]
                state[col_idx] = np.nan if np.isnan(prev) else y_hat - prev

        # ── Shift history buffer: y_hat becomes the new lag-1 ────────────
        history.insert(0, y_hat)    # O(N) but history is short (≤ 168)
        # Trim to avoid unbounded growth (keep only what we actually need)
        # The maximum window / lag is fixed; cap at len to save memory.
        # We do NOT pop here so history grows by exactly 1 per step and
        # naturally covers increasing lags as the rollout extends.

        # ── Lag features ─────────────────────────────────────────────────
        for col_idx, lag in lag_updates:
            if lag <= len(history):
                state[col_idx] = history[lag - 1]

        # ── Rolling features ─────────────────────────────────────────────
        _STAT_FNS = {
            "mean":   np.nanmean,
            "std":    np.nanstd,
            "min":    np.nanmin,
            "max":    np.nanmax,
            "median": np.nanmedian,
        }
        for col_idx, window, stat in roll_updates:
            window_vals = [v for v in history[:window] if not np.isnan(v)]
            if window_vals:
                fn = _STAT_FNS.get(stat, np.nanmean)
                state[col_idx] = fn(window_vals)
            # If all NaN, leave the existing value unchanged (best we can do)
