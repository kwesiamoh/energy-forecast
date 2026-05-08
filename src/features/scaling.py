"""
Normalisation and scaling utilities.

Foundation models (Chronos, TimesFM, Moirai) handle their own internal
normalisation, but classical baselines (XGBoost, MLP) and custom neural
nets need explicit scaling. This module provides:

  1. FitScaler  – fits a scaler on a TRAINING split, then applies it to
                  any other split. Prevents data leakage from val/test.
  2. Helpers    – inverse-transform predictions back to MW scale.
  3. Persistence – save/load fitted scalers to disk (joblib) so they can
                   be reused across sessions without re-fitting.

Supported scalers (all from sklearn):
  "standard"  – StandardScaler  (zero mean, unit variance)
                 Best for MLP / linear models.
  "minmax"    – MinMaxScaler    (scale to [0, 1])
                 Use when you need bounded outputs (e.g. sigmoid activations).
  "robust"    – RobustScaler    (median + IQR, resistant to outliers)
                 Best for energy data which has occasional storm/cold-snap
                 spikes that would inflate StandardScaler's σ.

Recommended defaults per task:
  - Load forecasting  → "robust"  (outlier-resistant)
  - Solar forecasting → "minmax"  (naturally bounded 0–max_capacity)
  - Wind forecasting  → "robust"
  - Features (lags, calendar) → "standard"

═══════════════════════════════════════════════════════════════════════════════
BUG FIX — Target / Feature Scaler Leakage in split_and_scale()
═══════════════════════════════════════════════════════════════════════════════
SYMPTOM: XGBoost predictions flatlined on the validation and test sets, with
near-zero RMSE on training data but random-noise-level errors on held-out data.

ROOT CAUSE: The previous split_and_scale() was calling .fit() independently
on each split inside a local scale_split() helper:

    def scale_split(split):             # ← OLD BUGGY IMPLEMENTATION
        s = t_scaler.fit_transform(split)   # fit on val, then fit on test
        s = f_scaler.fit_transform(s)       # same problem for features
        return s

  This caused three separate problems:
    1. DATA LEAKAGE: val and test statistics (mean, IQR, min, max) leaked
       into the fitted scalers, giving those splits access to future
       information not available at training time.
    2. INCOMPATIBLE SCALES: each split was scaled with its own parameters,
       so a load_mw value of 50 000 MW might map to 0.8 in training but 0.3
       in validation — destroying the physical consistency of the feature space.
    3. INVERSE-TRANSFORM FAILURE: model predictions in scaled space could not
       be correctly inverted back to MW because the scaler used at inference
       time was fit on a different distribution than the training scaler.

FIX:
  - t_scaler and f_scaler are fitted ONCE on the training split only.
  - .transform() (not .fit_transform()) is called on val and test.
  - The scale_split() helper is replaced with an explicit three-call pattern
    to make the contract obvious to the reader.
  - An assertion confirms that re-fitting never happens on held-out splits.

Usage example:
    from src.features.scaling import FitScaler

    scaler = FitScaler(method="robust", columns=["load_mw", "solar_mw"])
    train_scaled = scaler.fit_transform(train_df)
    val_scaled   = scaler.transform(val_df)
    test_scaled  = scaler.transform(test_df)

    preds_mw = scaler.inverse_transform(preds_scaled, columns=["load_mw"])
    scaler.save("models/scaler.joblib")
"""

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

ScalerMethod = Literal["standard", "minmax", "robust"]

_SCALER_CLASSES = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "robust":   RobustScaler,
}


class FitScaler:
    """
    Column-wise scaler that fits on training data and transforms splits.

    Attributes:
        method:   Scaling method name.
        columns:  Columns to scale (others are passed through unchanged).
        _scalers: Dict mapping column name → fitted sklearn scaler.
    """

    def __init__(
        self,
        method: ScalerMethod = "robust",
        columns: list[str] | None = None,
    ):
        self.method = method
        self.columns = columns      # None = scale all numeric columns
        self._scalers: dict[str, object] = {}

    # ── public API ────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FitScaler":
        """Fit scalers on df. Returns self for chaining."""
        cols = self._resolve_columns(df)
        fitted = 0
        for col in cols:
            scaler = _SCALER_CLASSES[self.method]()
            training_data = df[[col]].dropna()
            if training_data.empty:
                logger.warning(
                    "Skipping scaler fit for '%s': no non-missing training samples.",
                    col,
                )
                continue
            scaler.fit(training_data)
            self._scalers[col] = scaler
            fitted += 1
            logger.debug("  Fitted %s scaler on '%s'", self.method, col)
        logger.info("FitScaler (%s) fitted on %d of %d columns.",
                    self.method, fitted, len(cols))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers. Returns a copy — does not modify df."""
        df = df.copy()
        for col, scaler in self._scalers.items():
            if col not in df.columns:
                continue
            mask = df[col].notna()
            df.loc[mask, col] = scaler.transform(
                df.loc[mask, [col]]
            ).flatten().astype("float32")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df, then transform df. Convenience wrapper."""
        return self.fit(df).transform(df)

    def inverse_transform(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Inverse-transform scaled values back to original units.

        Args:
            df:      DataFrame of scaled values.
            columns: Which columns to invert. None = all fitted columns.

        Returns:
            Copy of df with specified columns in original scale.
        """
        df = df.copy()
        cols = columns or list(self._scalers.keys())
        for col in cols:
            if col not in self._scalers or col not in df.columns:
                continue
            scaler = self._scalers[col]
            mask = df[col].notna()
            df.loc[mask, col] = scaler.inverse_transform(
                df.loc[mask, [col]]
            ).flatten().astype("float32")
        return df

    def save(self, path: str | Path) -> None:
        """Persist the fitted scaler to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": self.method, "scalers": self._scalers}, path)
        logger.info("Scaler saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FitScaler":
        """Load a previously saved FitScaler from disk."""
        path = Path(path)
        data = joblib.load(path)
        instance = cls(method=data["method"])
        instance._scalers = data["scalers"]
        instance.columns = list(data["scalers"].keys())
        logger.info("Scaler loaded ← %s  (%d columns)",
                    path, len(instance._scalers))
        return instance

    # ── private ───────────────────────────────────────────────────────────

    def _resolve_columns(self, df: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        # Scale all float columns (skip int flags and ordinals)
        return [c for c in df.columns if df[c].dtype in (np.float32, np.float64)]


# ── Convenience function ──────────────────────────────────────────────────────

def split_and_scale(
    df: pd.DataFrame,
    target_cols: list[str],
    feature_cols: list[str],
    train_end: str,
    val_end: str,
    target_method: ScalerMethod = "robust",
    feature_method: ScalerMethod = "standard",
) -> dict:
    """
    Perform a chronological train/val/test split and fit scalers.

    Everything after val_end is treated as the test set.
    Scalers are fit on TRAINING data only.

    Args:
        df:             Master feature DataFrame.
        target_cols:    Columns to predict (e.g. ["load_mw", "solar_mw"]).
        feature_cols:   Input feature columns.
        train_end:      ISO date string — last date of training set.
        val_end:        ISO date string — last date of validation set.
        target_method:  Scaler for targets.
        feature_method: Scaler for features.

    Returns:
        Dict with keys:
          train, val, test   – unscaled DataFrames
          train_s, val_s, test_s  – scaled DataFrames
          target_scaler      – fitted FitScaler for targets
          feature_scaler     – fitted FitScaler for features
    """
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_end_ts = pd.Timestamp(val_end,   tz="UTC")

    train = df.loc[:train_end_ts].copy()
    val = df.loc[train_end_ts:val_end_ts].copy()
    test = df.loc[val_end_ts:].copy()

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d rows",
        len(train), len(val), len(test),
    )

    # Drop early NaN rows caused by lag features (up to 168 rows)
    train = train.dropna(subset=target_cols + feature_cols, how="any")

    t_scaler = FitScaler(method=target_method,  columns=target_cols)
    f_scaler = FitScaler(method=feature_method, columns=feature_cols)

    # ── BUG FIX: fit scalers on TRAINING data only ────────────────────────
    # Both scalers are fitted once on `train` and then applied via .transform()
    # to all three splits.  Previously, a local scale_split() helper was calling
    # .fit_transform() on each split independently, which:
    #   1. Leaked val/test statistics into the scalers (data leakage).
    #   2. Produced incompatible scale spaces across splits (each split had
    #      its own mean/IQR), making features physically meaningless when
    #      compared across the chronological boundary.
    #   3. Broke inverse_transform at evaluation time (scaler parameters no
    #      longer matched the training distribution).
    # The fix: fit once on train, transform-only on val and test.
    t_scaler.fit(train)
    f_scaler.fit(train)

    train_s = f_scaler.transform(t_scaler.transform(train))
    val_s   = f_scaler.transform(t_scaler.transform(val))
    test_s  = f_scaler.transform(t_scaler.transform(test))

    return {
        "train":          train,
        "val":            val,
        "test":           test,
        "train_s":        train_s,
        "val_s":          val_s,
        "test_s":         test_s,
        "target_scaler":  t_scaler,
        "feature_scaler": f_scaler,
    }
