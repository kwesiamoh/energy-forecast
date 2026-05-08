"""
Unified evaluation harness — all models, one leaderboard.

Loads every fitted model (SARIMA, XGBoost, Chronos zero-shot, Chronos
fine-tuned) and evaluates them against the same test split using the
same metrics.py harness. Produces a single ranked leaderboard table.

Usage:
    python -m src.models.evaluate_all              # run everything
    python -m src.models.evaluate_all --skip-sarima --quick
    python -m src.models.evaluate_all --target load_mw

The --quick flag runs Chronos inference on only the first 500 test rows
(useful for CPU-only machines where full inference is slow).

Outputs (written to results/):
    all_metrics.jsonl         – raw results, one JSON line per run
    leaderboard.csv           – ranked table (test set, h=1 MAE)
    leaderboard_plots/        – per-target bar charts + horizon curves
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.pipeline import TARGET_COLS, build_features, get_feature_cols
from src.features.scaling import split_and_scale
from src.models.metrics import MetricResult, ResultsRegistry, eval_by_horizon

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("results")
TRAIN_END     = "2021-12-31"
VAL_END       = "2022-12-31"
HORIZON       = 24


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all_evaluations(
    processed_dir: Path = PROCESSED_DIR,
    models_dir:    Path = MODELS_DIR,
    results_dir:   Path = RESULTS_DIR,
    train_end:     str  = TRAIN_END,
    val_end:       str  = VAL_END,
    horizon:       int  = HORIZON,
    targets:       list[str] = TARGET_COLS,
    skip_sarima:   bool = False,
    skip_chronos:  bool = False,
    quick:         bool = False,
) -> pd.DataFrame:
    """
    Evaluate all available models and return a leaderboard DataFrame.

    Models are evaluated in this order:
        1. XGBoost (baseline — always run)
        2. SARIMA  (baseline — skip with skip_sarima=True)
        3. Chronos zero-shot  (Phase 4 — skip with skip_chronos=True)
        4. Chronos fine-tuned (Phase 4 — only if checkpoint exists)

    Returns:
        DataFrame with all metrics, sorted by MAE ascending.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "leaderboard_plots").mkdir(exist_ok=True)
    registry = ResultsRegistry(results_dir / "all_metrics.jsonl")

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading feature dataset …")
    df        = build_features(processed_dir)
    feat_cols = get_feature_cols(df)
    splits    = split_and_scale(
        df, target_cols=targets, feature_cols=feat_cols,
        train_end=train_end, val_end=val_end,
    )
    val  = splits["val"]
    test = splits["test"]
    if quick:
        logger.info("--quick: truncating test set to 500 rows for speed.")
        test = test.iloc[:500]

    # ── 1. XGBoost ────────────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("Evaluating XGBoost …")
    _eval_xgboost(targets, models_dir, test, registry, horizon, val=val)

    # ── 2. SARIMA ─────────────────────────────────────────────────────────
    if not skip_sarima:
        logger.info("=" * 56)
        logger.info("Evaluating SARIMA …")
        _eval_sarima(targets, models_dir, test, registry, horizon, val=val)

    # ── 3. Chronos zero-shot ──────────────────────────────────────────────
    if not skip_chronos:
        logger.info("=" * 56)
        logger.info("Evaluating Chronos (zero-shot) …")
        _eval_chronos_zeroshot(targets, test, registry, horizon, val=val)

    # ── 4. Chronos fine-tuned ─────────────────────────────────────────────
    if not skip_chronos:
        logger.info("=" * 56)
        logger.info("Evaluating Chronos (fine-tuned) …")
        _eval_chronos_finetuned(targets, models_dir, test, registry, horizon, val=val)

    # ── Leaderboard ───────────────────────────────────────────────────────
    df_results = registry.to_dataframe()
    board      = _build_leaderboard(df_results, targets)

    csv_path   = results_dir / "leaderboard.csv"
    board.to_csv(csv_path, index=False)
    logger.info("Leaderboard saved → %s", csv_path)

    _print_leaderboard(board)
    _save_leaderboard_plots(df_results, targets, results_dir)

    return board


# ── Per-model evaluators ──────────────────────────────────────────────────────

def _eval_xgboost(targets, models_dir, test, registry, horizon, val=None):
    try:
        from src.models.xgboost_model import XGBoostForecaster
    except ImportError:
        logger.warning("xgboost not installed — skipping.")
        return

    baseline_dir = models_dir / "baselines"
    for target in targets:
        ckpt = baseline_dir / f"xgboost_{target}.meta.pkl"
        if not ckpt.exists():
            logger.warning("XGBoost checkpoint not found for %s — skipping.", target)
            continue
        model = XGBoostForecaster.load(baseline_dir / f"xgboost_{target}")
        result = model.evaluate(test, split_name="test", history_df=val)
        registry.add(result)
        _save_horizon_plot(model, test, target, "xgboost", horizon,
                           out_dir=Path("results/leaderboard_plots"))


def _eval_sarima(targets, models_dir, test, registry, horizon, val=None):
    try:
        from src.models.arima import SARIMAForecaster
        from src.models.metrics import MetricResult
    except ImportError:
        logger.warning("statsmodels not installed — skipping SARIMA.")
        return

    baseline_dir = models_dir / "baselines"
    for target in targets:
        ckpt = baseline_dir / f"sarima_{target}.pkl"
        if not ckpt.exists():
            logger.warning("SARIMA checkpoint not found for %s — skipping.", target)
            continue
        model   = SARIMAForecaster.load(ckpt)
        # Prepend val as warmup so the model isn't cold-starting on the test split
        combined = pd.concat([val, test]) if val is not None and len(val) > 0 else test
        preds    = model.predict_rolling(combined)
        n_warmup = len(val) if val is not None else 0
        preds    = preds[n_warmup:]     # discard warmup rows
        y_true   = test[target].values
        result   = MetricResult.from_arrays(
            model="sarima", target=target, split="test",
            horizon=horizon, y_true=y_true, y_pred=preds[:, 0],
        )
        registry.add(result)


def _eval_chronos_zeroshot(targets, test, registry, horizon, val=None):
    try:
        from src.models.chronos_model import ChronosForecaster
    except ImportError:
        logger.warning("amazon-chronos-t5 not installed — skipping Chronos.")
        return

    for target in targets:
        logger.info("  Chronos zero-shot: %s", target)
        model = ChronosForecaster(target_col=target, horizon=horizon)
        # predict() loads the pipeline lazily on first call
        result = model.evaluate(test, split_name="test", history_df=val)
        registry.add(result)


def _eval_chronos_finetuned(targets, models_dir, test, registry, horizon, val=None):
    try:
        from src.models.chronos_model import ChronosForecaster
    except ImportError:
        return

    ft_dir = models_dir / "chronos_finetuned"
    for target in targets:
        ckpt = ft_dir / target
        if not (ckpt / "meta.pkl").exists():
            logger.info("  No fine-tuned Chronos checkpoint for %s — skipping.", target)
            continue
        logger.info("  Chronos fine-tuned: %s", target)
        model  = ChronosForecaster.load(ckpt)
        result = model.evaluate(test, split_name="test", history_df=val)
        registry.add(result)


# ── Leaderboard table ─────────────────────────────────────────────────────────

def _build_leaderboard(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Pivot and rank all test-set results by MAE."""
    test_df = df[df["split"] == "test"].copy()
    cols    = ["model", "target", "split", "mae", "rmse", "mape", "smape", "r2", "nrmse"]
    cols    = [c for c in cols if c in test_df.columns]
    board   = test_df[cols].sort_values(["target", "mae"]).reset_index(drop=True)
    return board


def _print_leaderboard(board: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print(f"{'MODEL':<22} {'TARGET':<22} {'MAE':>8} {'RMSE':>8} {'MAPE%':>7} {'R²':>7}")
    print("-" * 74)
    prev_target = None
    for _, row in board.iterrows():
        if row["target"] != prev_target and prev_target is not None:
            print()
        prev_target = row["target"]
        print(
            f"{row['model']:<22} {row['target']:<22} "
            f"{row['mae']:>8.1f} {row['rmse']:>8.1f} "
            f"{row.get('mape', float('nan')):>7.2f} "
            f"{row.get('r2',   float('nan')):>7.4f}"
        )
    print("=" * 74 + "\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_leaderboard_plots(df: pd.DataFrame, targets: list[str], out_dir: Path) -> None:
    """Bar chart: MAE per model per target, test set."""
    test_df = df[df["split"] == "test"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ax, target in zip(axes.flat, targets):
        sub     = test_df[test_df["target"] == target].sort_values("mae")
        colors  = ["#1D9E75" if "xgboost" in m else
                   "#7F77DD" if "chronos" in m else
                   "#BA7517" for m in sub["model"]]
        bars    = ax.barh(sub["model"], sub["mae"], color=colors, height=0.5)
        ax.set_xlabel("MAE [MW]")
        ax.set_title(target)
        ax.invert_yaxis()

        # Label values on bars
        for bar in bars:
            w = bar.get_width()
            ax.text(w + w * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}", va="center", fontsize=9)

    plt.suptitle("Model leaderboard — MAE on test set (lower is better)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "leaderboard_mae_bar.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Leaderboard bar chart → %s", path)


def _save_horizon_plot(model, test, target, model_label, horizon, out_dir):
    """Per-horizon MAE curve for a single model."""
    try:
        preds  = model.predict(test)
        y_true = test[target].values
        rows   = []
        for h in range(min(horizon, preds.shape[1])):
            yp = preds[:, h]
            if h == 0:
                yt = y_true.copy()
            else:
                yt = np.full_like(y_true, np.nan)
                yt[:-h] = y_true[h:]
            mask = np.isfinite(yt) & np.isfinite(yp)
            rows.append({"h": h + 1, "mae": float(np.mean(np.abs(yt[mask] - yp[mask])))})

        hdf = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(hdf["h"], hdf["mae"], marker="o", ms=3, color="steelblue")
        ax.set_xlabel("Horizon h"); ax.set_ylabel("MAE [MW]")
        ax.set_title(f"{model_label} — {target} horizon curve (test)")
        plt.tight_layout()
        p = out_dir / f"{model_label}_{target}_horizon.png"
        plt.savefig(p, dpi=110, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning("Horizon plot failed for %s/%s: %s", model_label, target, e)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full model leaderboard evaluation.")
    parser.add_argument("--skip-sarima",  action="store_true")
    parser.add_argument("--skip-chronos", action="store_true")
    parser.add_argument("--quick",        action="store_true", help="500-row test subset")
    parser.add_argument("--target",       nargs="+", default=TARGET_COLS)
    parser.add_argument("--train-end",    default=TRAIN_END)
    parser.add_argument("--val-end",      default=VAL_END)
    parser.add_argument("--horizon",      type=int, default=HORIZON)
    args = parser.parse_args()

    run_all_evaluations(
        skip_sarima=args.skip_sarima,
        skip_chronos=args.skip_chronos,
        quick=args.quick,
        targets=args.target,
        train_end=args.train_end,
        val_end=args.val_end,
        horizon=args.horizon,
    )
