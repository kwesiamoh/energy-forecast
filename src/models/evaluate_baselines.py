"""
Baseline evaluation harness.

Runs SARIMA and XGBoost against all four targets, collects metrics into
a ResultsRegistry, and produces publication-ready comparison tables and
plots. This is the script you run to generate the benchmark numbers that
all future Phase 4 (foundation) models must beat.

Usage (from repo root):
    python -m src.models.evaluate_baselines

Or import and call run_baseline_evaluation() from a notebook.

Outputs (written to results/):
    baseline_metrics.jsonl  – raw metrics (one JSON line per model×target×split)
    baseline_table.csv      – formatted comparison table
    baseline_plots/         – per-target forecast vs actual plots
                              + feature importance bar charts
                              + horizon error curves
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.pipeline import TARGET_COLS, build_features, get_feature_cols
from src.features.scaling import split_and_scale
from src.models.arima import SARIMAForecaster
from src.models.metrics import MetricResult, ResultsRegistry, eval_by_horizon
from src.models.xgboost_model import XGBoostForecaster

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")
MODELS_DIR    = Path("models/baselines")

TRAIN_END = "2021-12-31"
VAL_END   = "2022-12-31"
# Everything after VAL_END is the test set

HORIZON = 24   # hours ahead

# Set to False to skip SARIMA (much slower than XGBoost)
RUN_SARIMA = True

# SARIMA: set auto_order=False for a quick run using fixed (2,1,2)(1,1,1,24)
SARIMA_AUTO_ORDER = True


# ── Main entry point ──────────────────────────────────────────────────────────

def run_baseline_evaluation(
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path   = RESULTS_DIR,
    models_dir: Path    = MODELS_DIR,
    train_end: str      = TRAIN_END,
    val_end: str        = VAL_END,
    horizon: int        = HORIZON,
    run_sarima: bool    = RUN_SARIMA,
    sarima_auto: bool   = SARIMA_AUTO_ORDER,
    targets: list[str]  = TARGET_COLS,
) -> pd.DataFrame:
    """
    Full baseline evaluation pipeline.

    Returns:
        DataFrame with one row per (model, target, split) combination.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "baseline_plots").mkdir(exist_ok=True)

    registry = ResultsRegistry(results_dir / "baseline_metrics.jsonl")

    # ── 1. Load features ─────────────────────────────────────────────────
    logger.info("Loading feature dataset …")
    df = build_features(processed_dir)
    feat_cols = get_feature_cols(df)

    splits = split_and_scale(
        df,
        target_cols=targets,
        feature_cols=feat_cols,
        train_end=train_end,
        val_end=val_end,
    )
    train = splits["train"]
    val   = splits["val"]
    test  = splits["test"]

    logger.info(
        "Splits — train: %d, val: %d, test: %d rows",
        len(train), len(val), len(test),
    )

    # ── 2. XGBoost ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 1: XGBoost (direct multi-step)")
    logger.info("=" * 60)

    for target in targets:
        logger.info("  Target: %s", target)

        xgb_model = XGBoostForecaster(
            target_col=target,
            horizon=horizon,
        )
        xgb_model.fit(train, val_df=val)
        xgb_model.save(models_dir / f"xgboost_{target}")

        for split_name, split_df, history in [
            ("val",  val,  None),   # val has train as warmup — already in model state
            ("test", test, val),    # test cold-starts at 2023-01-01; pass val as warmup
        ]:
            result = xgb_model.evaluate(split_df, split_name=split_name, history_df=history)
            registry.add(result)

            # Horizon error curve
            preds = xgb_model.predict(split_df)
            _save_horizon_plot(
                preds, split_df[target].values,
                model_name="xgboost", target=target,
                split=split_name, horizon=horizon,
                out_dir=results_dir / "baseline_plots",
            )

        # Feature importance plot (h=1)
        _save_importance_plot(
            xgb_model.feature_importance(horizon_step=1),
            model_name="xgboost", target=target,
            out_dir=results_dir / "baseline_plots",
        )

        # Forecast vs actual (test set, first 7 days)
        _save_forecast_plot(
            preds[:168, 0],
            test[target].values[:168],
            model_name="xgboost", target=target,
            out_dir=results_dir / "baseline_plots",
        )

    # ── 3. SARIMA ────────────────────────────────────────────────────────
    if run_sarima:
        logger.info("=" * 60)
        logger.info("BASELINE 2: SARIMA (univariate recursive)")
        logger.info("=" * 60)

        for target in targets:
            logger.info("  Target: %s", target)

            sarima = SARIMAForecaster(
                target_col=target,
                horizon=horizon,
                auto_order=sarima_auto,
            )
            sarima.fit(train, val_df=val)
            sarima.save(models_dir / f"sarima_{target}.pkl")

            for split_name, split_df, history in [
                ("val",  val,  None),
                ("test", test, val),
            ]:
                # SARIMA predict is slow for large test sets — use rolling approx
                preds = sarima.predict_rolling(
                    pd.concat([history, split_df]) if history is not None else split_df
                )
                # Discard warmup rows so metrics cover only the target split
                if history is not None:
                    preds = preds[len(history):]
                y_true = split_df[target].values

                result = MetricResult.from_arrays(
                    model=sarima.model_name,
                    target=target,
                    split=split_name,
                    horizon=horizon,
                    y_true=y_true,
                    y_pred=preds[:, 0],
                )
                registry.add(result)

                _save_horizon_plot(
                    preds, y_true,
                    model_name="sarima", target=target,
                    split=split_name, horizon=horizon,
                    out_dir=results_dir / "baseline_plots",
                )

    # ── 4. Summary table ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    summary = registry.to_dataframe()
    _print_leaderboard(summary, targets)

    table_path = results_dir / "baseline_table.csv"
    summary.to_csv(table_path, index=False)
    logger.info("Full results saved → %s", table_path)

    return summary


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save_forecast_plot(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    model_name: str,
    target: str,
    out_dir: Path,
    n_hours: int = 168,
) -> None:
    """Forecast vs actual — first n_hours of the test set."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true[:n_hours],  label="Actual",    lw=1.5, color="steelblue")
    ax.plot(y_pred[:n_hours],  label=model_name,  lw=1.2, color="crimson", alpha=0.85)
    ax.set_title(f"{model_name.upper()} — {target} (first {n_hours}h of test set)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MW")
    ax.legend()
    plt.tight_layout()
    path = out_dir / f"{model_name}_{target}_forecast.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved forecast plot → %s", path)


def _save_horizon_plot(
    preds: np.ndarray,
    y_true: np.ndarray,
    model_name: str,
    target: str,
    split: str,
    horizon: int,
    out_dir: Path,
) -> None:
    """MAE and RMSE vs forecast horizon (h=1…H)."""
    if preds.ndim == 1:
        return  # can't compute horizon curve for 1-D output

    rows = []
    for h in range(min(horizon, preds.shape[1])):
        yp = preds[:, h]
        yt = y_true
        # Shift true values to align with h-step-ahead predictions
        if h > 0:
            yt = np.roll(y_true, -h)
            yt[-h:] = np.nan
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() == 0:
            continue
        rows.append({
            "horizon": h + 1,
            "mae":  float(np.mean(np.abs(yt[mask] - yp[mask]))),
            "rmse": float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2))),
        })

    if not rows:
        return

    hdf = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax1.plot(hdf["horizon"], hdf["mae"],  marker="o", ms=3, color="steelblue")
    ax1.set_title("MAE by horizon")
    ax1.set_xlabel("h (hours ahead)"); ax1.set_ylabel("MAE [MW]")

    ax2.plot(hdf["horizon"], hdf["rmse"], marker="o", ms=3, color="crimson")
    ax2.set_title("RMSE by horizon")
    ax2.set_xlabel("h (hours ahead)"); ax2.set_ylabel("RMSE [MW]")

    plt.suptitle(f"{model_name.upper()} — {target} [{split}]", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{model_name}_{target}_{split}_horizon.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved horizon plot → %s", path)


def _save_importance_plot(
    importance: pd.Series,
    model_name: str,
    target: str,
    out_dir: Path,
    top_n: int = 25,
) -> None:
    """Horizontal bar chart of top-N feature importances."""
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
    top.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Gain")
    ax.set_title(f"{model_name.upper()} feature importance — {target} (h=1)")
    ax.invert_yaxis()
    plt.tight_layout()
    path = out_dir / f"{model_name}_{target}_importance.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved importance plot → %s", path)


def _print_leaderboard(df: pd.DataFrame, targets: list[str]) -> None:
    """Print a formatted leaderboard table to stdout."""
    print("\n" + "=" * 72)
    print(f"{'MODEL':<12} {'TARGET':<22} {'SPLIT':<6} "
          f"{'MAE':>8} {'RMSE':>8} {'MAPE%':>7} {'R²':>7}")
    print("-" * 72)

    test_rows = df[df["split"] == "test"].sort_values(["target", "mae"])
    for _, row in test_rows.iterrows():
        print(
            f"{row['model']:<12} {row['target']:<22} {row['split']:<6} "
            f"{row['mae']:>8.1f} {row['rmse']:>8.1f} "
            f"{row.get('mape', float('nan')):>7.2f} "
            f"{row.get('r2', float('nan')):>7.4f}"
        )
    print("=" * 72 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline model evaluation.")
    parser.add_argument("--no-sarima",  action="store_true", help="Skip SARIMA (faster)")
    parser.add_argument("--no-auto",    action="store_true", help="Use fixed SARIMA order")
    parser.add_argument("--horizon",    type=int, default=24, help="Forecast horizon (hours)")
    parser.add_argument("--targets",    nargs="+", default=TARGET_COLS)
    parser.add_argument("--train-end",  default=TRAIN_END)
    parser.add_argument("--val-end",    default=VAL_END)
    args = parser.parse_args()

    run_baseline_evaluation(
        run_sarima=not args.no_sarima,
        sarima_auto=not args.no_auto,
        horizon=args.horizon,
        targets=args.targets,
        train_end=args.train_end,
        val_end=args.val_end,
    )
