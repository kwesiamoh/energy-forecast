"""Canonical six-model evaluation runner and h=1 leaderboard."""

import argparse
import gc
import logging
import math
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.features.pipeline import TARGET_COLS, build_features, get_feature_cols
from src.features.scaling import split_and_scale
from src.models.foundation_utils import (
    build_context_actuals,
    foundation_input_fingerprint,
)
from src.models.metrics import (
    MetricResult,
    ResultsRegistry,
    eval_by_horizon,
    get_target_unit,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
TRAIN_END = "2021-12-31"
VAL_END = "2022-12-31"
HORIZON = 24
CONTEXT_LENGTH = 168
QUICK_ROWS = 500
SARIMA_TARGETS = ['load_mw']

MODEL_CHOICES = [
    "seasonal_naive_24",
    "seasonal_naive_168",
    "sarima",
    "xgboost",
    "chronos_t5",
    "timesfm",
    "moirai2",
]


def _target_benchmark_fingerprint(target, val, test, horizon, quick):
    contexts, actuals = build_context_actuals(
        val[target].tail(CONTEXT_LENGTH).to_numpy(),
        test[target].to_numpy(),
        CONTEXT_LENGTH,
        horizon,
    )
    return foundation_input_fingerprint(
        target=target,
        origins_ns=test.index.asi8,
        contexts=contexts,
        actuals=actuals,
        benchmark_set='final_six_v1',
        run_mode='quick' if quick else 'full',
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
    )


def run_all_evaluations(
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
    horizon: int = HORIZON,
    targets: list[str] = TARGET_COLS,
    models: list[str] | None = None,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 16,
    quick: bool = False,
    skip_sarima: bool = False,
    skip_chronos: bool = False,
) -> pd.DataFrame:
    """Evaluate selected models with one forecast contract and metric path."""
    selected = list(models or MODEL_CHOICES)
    unknown = sorted(set(selected) - set(MODEL_CHOICES))
    if unknown:
        raise ValueError(f"Unknown model selection: {', '.join(unknown)}")
    if skip_sarima:
        selected = [name for name in selected if name != "sarima"]
    if skip_chronos:
        selected = [name for name in selected if name != "chronos_t5"]

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = results_dir / "leaderboard_plots"
    plot_dir.mkdir(exist_ok=True)
    registry = ResultsRegistry(results_dir / "all_metrics.jsonl")

    logger.info("Loading feature dataset ...")
    df = build_features(processed_dir)
    feature_cols = get_feature_cols(df, train_end=train_end)
    splits = split_and_scale(
        df,
        target_cols=targets,
        feature_cols=feature_cols,
        train_end=train_end,
        val_end=val_end,
    )
    val = splits["val"]
    test = splits["test"]
    if quick:
        logger.info(
            "QUICK mode: evaluating the first %d test origins only.", QUICK_ROWS
        )
        test = test.iloc[:QUICK_ROWS]
    else:
        logger.info("FULL mode: evaluating the complete configured test period.")

    for model_key in MODEL_CHOICES:
        if model_key not in selected:
            continue
        logger.info("=" * 60)
        logger.info("Evaluating %s", model_key)
        if model_key.startswith("seasonal_naive_"):
            season = int(model_key.rsplit("_", 1)[1])
            _eval_seasonal_naive(
                season, targets, val, test, registry, horizon, plot_dir, quick
            )
        elif model_key == "sarima":
            _eval_sarima(
                targets, models_dir, val, test, registry, horizon, quick
            )
        elif model_key == "xgboost":
            _eval_xgboost(
                targets, models_dir, val, test, registry, horizon, plot_dir, quick
            )
        else:
            _eval_foundation(
                model_key=model_key,
                targets=targets,
                val=val,
                test=test,
                registry=registry,
                horizon=horizon,
                plot_dir=plot_dir,
                quick=quick,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )
            gc.collect()

    all_results = registry.to_dataframe()
    board = _build_leaderboard(all_results, quick=quick)
    suffix = "_quick" if quick else ""
    csv_path = results_dir / f"leaderboard{suffix}.csv"
    board.to_csv(csv_path, index=False)
    logger.info("Leaderboard saved -> %s", csv_path)
    _print_leaderboard(board)
    _save_leaderboard_plots(board, targets, results_dir, quick=quick)
    return board


def _eval_seasonal_naive(
    season, targets, val, test, registry, horizon, plot_dir, quick
):
    from src.models.seasonal_naive import SeasonalNaiveForecaster

    for target in targets:
        model = SeasonalNaiveForecaster(
            target_col=target, horizon=horizon, season_length=season
        )
        result, preds = _evaluate_predictions(
            model, target, val, test, horizon, quick, history_length=season
        )
        registry.add(result)
        _save_horizon_plot(
            preds, test[target].to_numpy(), model.model_name, target, plot_dir
        )


def _eval_xgboost(
    targets, models_dir, val, test, registry, horizon, plot_dir, quick
):
    try:
        from src.models.xgboost_model import XGBoostForecaster
    except ImportError:
        logger.warning("xgboost is not installed; skipping XGBoost.")
        return

    baseline_dir = models_dir / "baselines"
    for target in targets:
        checkpoint = baseline_dir / f"xgboost_{target}.meta.pkl"
        if not checkpoint.exists():
            logger.warning("XGBoost checkpoint not found for %s; skipping.", target)
            continue
        model = XGBoostForecaster.load(baseline_dir / f"xgboost_{target}")
        started = time.perf_counter()
        combined = pd.concat([val, test])
        preds = model.predict(combined)[len(val):]
        result = MetricResult.from_arrays(
            model=model.model_name,
            target=target,
            split="test",
            horizon=horizon,
            y_true=test[target].to_numpy(),
            y_pred=preds[:, 0],
            quick=quick,
            run_mode="quick" if quick else "full",
            elapsed_evaluation_seconds=time.perf_counter() - started,
            benchmark_set="final_six_v1",
            input_fingerprint=_target_benchmark_fingerprint(
                target, val, test, horizon, quick
            ),
        )
        registry.add(result)
        _save_horizon_plot(
            preds, test[target].to_numpy(), model.model_name, target, plot_dir
        )


def _eval_sarima(
    targets, models_dir, val, test, registry, horizon, quick
):
    try:
        from src.models.arima import SARIMAForecaster
    except ImportError:
        logger.warning("statsmodels/pmdarima is not installed; skipping SARIMA.")
        return

    baseline_dir = models_dir / "baselines"
    for target in (target for target in SARIMA_TARGETS if target in targets):
        checkpoint = baseline_dir / f"sarima_{target}.pkl"
        if not checkpoint.exists():
            logger.warning("SARIMA checkpoint not found for %s; skipping.", target)
            continue
        model = SARIMAForecaster.load(checkpoint)
        combined = pd.concat([val, test])
        started = time.perf_counter()
        preds = model.predict_rolling(combined)[len(val):]
        result = MetricResult.from_arrays(
            model=model.model_name,
            target=target,
            split="test",
            horizon=horizon,
            y_true=test[target].to_numpy(),
            y_pred=preds[:, 0],
            quick=quick,
            run_mode="quick" if quick else "full",
            elapsed_evaluation_seconds=time.perf_counter() - started,
            benchmark_set="final_six_v1",
            input_fingerprint=_target_benchmark_fingerprint(
                target, val, test, horizon, quick
            ),
        )
        registry.add(result)


def _eval_foundation(
    model_key,
    targets,
    val,
    test,
    registry,
    horizon,
    plot_dir,
    quick,
    device,
    dtype,
    batch_size,
):
    try:
        if model_key == "chronos_t5":
            from src.models.chronos_model import ChronosForecaster

            model_class = ChronosForecaster
        elif model_key == "timesfm":
            from src.models.timesfm_model import TimesFMForecaster

            model_class = TimesFMForecaster
        else:
            from src.models.moirai_model import MoiraiForecaster

            model_class = MoiraiForecaster
    except ImportError as exc:
        logger.warning("%s unavailable: %s", model_key, exc)
        return

    try:
        model = model_class(
            target_col=targets[0],
            horizon=horizon,
            context_length=CONTEXT_LENGTH,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
    except ImportError as exc:
        logger.warning("%s unavailable: %s", model_key, exc)
        return
    for target in targets:
        logger.info("  %s: %s", model_key, target)
        model.target_col = target
        try:
            result, preds = _evaluate_predictions(
                model,
                target,
                val,
                test,
                horizon,
                quick,
                history_length=CONTEXT_LENGTH,
            )
        except ImportError as exc:
            logger.warning("%s unavailable: %s", model_key, exc)
            break

        result.meta.update(
            {
                "device": model.device,
                "gpu_name": model._device_spec.gpu_name,
                "dtype": model.dtype,
                "batch_size": model.batch_size,
                "context_length": model.context_length,
                "inference_seconds": model.last_runtime_seconds,
            }
        )
        registry.add(result)
        _save_horizon_plot(
            preds, test[target].to_numpy(), model.model_name, target, plot_dir
        )
    del model


def _evaluate_predictions(
    model, target, val, test, horizon, quick, history_length
):
    """Run one model once and return its h=1 metric plus all horizon columns."""
    history = val.tail(history_length)
    combined = pd.concat([history, test])
    started = time.perf_counter()
    all_predictions = model.predict(combined)
    predictions = all_predictions[len(history):]
    elapsed = time.perf_counter() - started
    result = MetricResult.from_arrays(
        model=model.model_name,
        target=target,
        split="test",
        horizon=horizon,
        y_true=test[target].to_numpy(),
        y_pred=predictions[:, 0],
        quick=quick,
        run_mode="quick" if quick else "full",
        elapsed_evaluation_seconds=elapsed,
        benchmark_set="final_six_v1",
        input_fingerprint=_target_benchmark_fingerprint(
            target, val, test, horizon, quick
        ),
    )
    return result, predictions


def _build_leaderboard(df: pd.DataFrame, quick: bool) -> pd.DataFrame:
    """Rank only results from the requested run mode by horizon-1 MAE."""
    if df.empty:
        return df
    test_df = df[df["split"] == "test"].copy()
    if "meta" in test_df:
        current_benchmark = test_df["meta"].map(
            lambda value: isinstance(value, dict)
            and value.get("benchmark_set") == "final_six_v1"
        )
        test_df = test_df[current_benchmark]
    if "quick" in test_df:
        mode = test_df["quick"].fillna(False).astype(bool)
        test_df = test_df[mode if quick else ~mode]
    test_df = test_df[
        (test_df['model'] != 'sarima')
        | test_df['target'].isin(SARIMA_TARGETS)
    ]
    columns = [
        "model",
        "target",
        "split",
        "evaluated_step",
        "sample_count",
        "quick",
        "mae",
        "rmse",
        "mape",
        "smape",
        "r2",
        "nrmse",
        "meta",
    ]
    columns = [column for column in columns if column in test_df.columns]
    return test_df[columns].sort_values(["target", "mae"]).reset_index(drop=True)


def _print_leaderboard(board: pd.DataFrame) -> None:
    print("\n" + "=" * 76)
    print(
        f"{'MODEL':<25} {'TARGET':<22} {'MAE h=1':>9} "
        f"{'RMSE h=1':>10} {'R2 h=1':>8}"
    )
    print("-" * 76)
    for _, row in board.iterrows():
        print(
            f"{row['model']:<25} {row['target']:<22} "
            f"{row['mae']:>9.1f} {row['rmse']:>10.1f} {row['r2']:>8.4f}"
        )
    print("=" * 76 + "\n")


def _save_leaderboard_plots(
    board: pd.DataFrame, targets: list[str], out_dir: Path, quick: bool
) -> None:
    if board.empty:
        return
    columns = 3
    rows = math.ceil(len(targets) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
    axes = list(getattr(axes, "flat", [axes]))
    color_map = {
        "seasonal_naive_24": "#888888",
        "seasonal_naive_168": "#B8B8B8",
        "sarima": "#BA7517",
        "xgboost": "#1D9E75",
        "chronos_t5_small": "#7F77DD",
        "timesfm_2_5": "#D85A30",
        "moirai_2_0_small": "#378ADD",
    }
    for axis, target in zip(axes, targets):
        subset = board[board["target"] == target].sort_values("mae")
        bars = axis.barh(
            subset["model"],
            subset["mae"],
            color=[color_map.get(name, "#777777") for name in subset["model"]],
        )
        axis.set_xlabel(f"Horizon-1 MAE [{get_target_unit(target)}]")
        axis.set_title(target)
        axis.invert_yaxis()
        for bar in bars:
            width = bar.get_width()
            axis.text(
                width * 1.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}",
                va="center",
                fontsize=8,
            )
    for axis in axes[len(targets):]:
        axis.set_visible(False)
    mode = "QUICK subset" if quick else "FULL test set"
    fig.suptitle(f"Model leaderboard: horizon-1 MAE ({mode})")
    fig.tight_layout()
    suffix = "_quick" if quick else ""
    path = out_dir / f"leaderboard_mae_bar{suffix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_horizon_plot(
    predictions, actuals, model_name, target, out_dir
) -> None:
    horizon_metrics = eval_by_horizon(actuals, predictions)
    fig, axis = plt.subplots(figsize=(8, 3))
    axis.plot(
        horizon_metrics["horizon"],
        horizon_metrics["mae"],
        marker="o",
        markersize=3,
    )
    axis.set_xlabel("Forecast horizon h=1..24")
    axis.set_ylabel(f"MAE [{get_target_unit(target)}]")
    axis.set_title(f"{model_name}: {target} horizon-wise MAE")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{model_name}_{target}_horizon.png",
        dpi=110,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the canonical model benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=MODEL_CHOICES,
        help="Models to evaluate.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-mode", choices=["quick", "full"], default="full")
    parser.add_argument(
        "--quick", action="store_true", help="Alias for --run-mode quick."
    )
    parser.add_argument("--skip-sarima", action="store_true")
    parser.add_argument("--skip-chronos", action="store_true")
    parser.add_argument("--target", nargs="+", default=TARGET_COLS)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--val-end", default=VAL_END)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()

    run_all_evaluations(
        models=args.models,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        quick=args.quick or args.run_mode == "quick",
        skip_sarima=args.skip_sarima,
        skip_chronos=args.skip_chronos,
        targets=args.target,
        train_end=args.train_end,
        val_end=args.val_end,
        horizon=args.horizon,
    )
