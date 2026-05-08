"""
Chronos foundation model wrapper.

Chronos (Amazon, Apache 2.0) is a family of pretrained time-series
foundation models based on the T5 architecture. It tokenises time-series
values into a fixed vocabulary and generates probabilistic forecasts
autoregressively — no features needed, just past values.

Paper: "Chronos: Learning the Language of Time Series" (Ansari et al. 2024)
Repo:  https://github.com/amazon-science/chronos-forecasting
Install: pip install amazon-chronos-t5 transformers accelerate

Model variants (smallest → largest, speed ↔ accuracy tradeoff):
    chronos-t5-tiny    ~8M  params   fastest, good for prototyping
    chronos-t5-mini    ~20M params
    chronos-t5-small   ~46M params   recommended starting point
    chronos-t5-base    ~200M params
    chronos-t5-large   ~710M params  best accuracy, needs GPU

Forecasting strategy
────────────────────
Chronos is a CONTEXT → FORECAST model:
  - It receives the last `context_length` values of the target series
  - It autoregressively generates `horizon` steps ahead
  - Outputs are probabilistic: we use the median as the point forecast
    and optionally store quantiles for uncertainty intervals

This wrapper implements TWO modes:

  1. Zero-shot (fit() is a no-op):
     Load a pretrained checkpoint and call predict() immediately.
     No training data needed. This is the "free lunch" test —
     if Chronos zero-shot beats XGBoost, the features aren't helping.

  2. Fine-tuned:
     Call fit() to continue training the model on your OPSD series.
     Uses the chronos fine-tuning script approach: converts the target
     column into ChronosDataset format and runs a training loop with
     AdamW + cosine LR schedule. Early stopping on val NLL.

Context window strategy
───────────────────────
Chronos was pretrained with context up to 512 tokens. For hourly energy
data we use context_length=168 (1 week) as a strong default — this
captures the weekly seasonal cycle which is the dominant pattern.
You can experiment with 336 (2 weeks) or 720 (30 days) if GPU memory
allows, but diminishing returns beyond 168h for most energy targets.

═══════════════════════════════════════════════════════════════════════════════
BUG FIX — Sequence Contiguity (non-contiguous DatetimeIndex)
═══════════════════════════════════════════════════════════════════════════════
SYMPTOM:
  The feature pipeline drops rows with missing exogenous features, leaving gaps
  in the DatetimeIndex passed to fit().  _make_sliding_windows maps array
  position directly to time step, so a gap of N hours would be silently treated
  as N consecutive 1-hour steps, misaligning every subsequent context/forecast
  pair and corrupting the temporal patterns learned during fine-tuning.

FIX:
  fit() now reindexes train_df and val_df onto a contiguous pd.date_range at
  "h" frequency before extracting the target series.  Gap positions become NaN.
  _make_sliding_windows already skips any window whose forecast target contains
  NaN (guard: `np.isnan(tgt_win).any()`), so no additional filtering is needed.
  Context windows that contain NaN are passed through as-is; Chronos treats
  NaN as missing and pads internally.

Batched inference
─────────────────
predict() batches all forecast origins into a single forward pass using
ChronosPipeline.predict(). This is orders of magnitude faster than
looping row-by-row and is the correct way to use the Chronos API.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .base import BaseForecaster

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "amazon/chronos-t5-small"
DEFAULT_CONTEXT = 168    # hours (1 week) — covers daily + weekly cycles
DEFAULT_SAMPLES = 20     # Monte Carlo samples for median/quantile estimation
FINETUNE_DEFAULTS = {
    "learning_rate":    1e-4,
    "num_epochs":       10,
    "batch_size":       32,
    "lr_scheduler":     "cosine",
    "warmup_steps":     100,
    "weight_decay":     1e-2,
    "early_stopping_patience": 3,   # epochs without val improvement
}


class ChronosForecaster(BaseForecaster):
    """
    Chronos T5 foundation model — zero-shot and fine-tuned modes.

    Args:
        target_col:      Column in the feature DataFrame to forecast.
        horizon:         Forecast horizon in hours.
        model_name:      HuggingFace model ID or local checkpoint path.
        context_length:  Number of past hours fed as context to the model.
        num_samples:     Monte Carlo samples (higher = better quantiles, slower).
        device:          "cuda", "mps", or "cpu". Auto-detected if None.
        finetune_config: Dict of fine-tuning hyperparameters (see FINETUNE_DEFAULTS).
        quantile_levels: Quantile levels to store in addition to median.
                         e.g. [0.1, 0.9] gives an 80% prediction interval.
    """

    model_name_str = "chronos"  # used in MetricResult / leaderboard

    # BaseForecaster uses model_name as a class attribute — override it
    @property
    def model_name(self) -> str:  # type: ignore[override]
        suffix = "ft" if self._is_finetuned else "zs"
        return f"chronos-{suffix}"

    def __init__(
        self,
        target_col:      str = "load_mw",
        horizon:         int = 24,
        model_id:        str = DEFAULT_MODEL,
        context_length:  int = DEFAULT_CONTEXT,
        num_samples:     int = DEFAULT_SAMPLES,
        device:          str | None = None,
        finetune_config: dict | None = None,
        quantile_levels: list[float] | None = None,
    ):
        super().__init__(target_col=target_col, horizon=horizon)
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self.device = device or _auto_device()
        self.finetune_config = {**FINETUNE_DEFAULTS, **(finetune_config or {})}
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        self._pipeline = None   # ChronosPipeline — loaded lazily
        self._is_finetuned = False
        self.is_fitted = True
        self._ft_checkpoint = None   # path to fine-tuned weights

    # ── Lazy pipeline loading ─────────────────────────────────────────────

    def _load_pipeline(self, checkpoint: str | Path | None = None) -> None:
        """Load (or reload) the ChronosPipeline from a checkpoint."""
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "Install Chronos: pip install amazon-chronos-t5 transformers accelerate"
            )

        source = str(checkpoint) if checkpoint else self.model_id
        self._logger.info(
            "Loading ChronosPipeline from '%s' on %s …", source, self.device)

        self._pipeline = ChronosPipeline.from_pretrained(
            source,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        )
        self._logger.info("Pipeline loaded.")

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame | None = None,
    ) -> "ChronosForecaster":
        """
        Fine-tune Chronos on the target series from train_df.

        If you just want zero-shot, skip fit() entirely and call predict()
        directly — the pipeline will be loaded on first predict() call.

        Fine-tuning approach:
          - Converts train_df[target_col] into overlapping context/forecast
            pairs using a sliding window of size context_length + horizon.
          - Trains using AdamW with cosine LR decay.
          - Early stops on val NLL if val_df is provided.
          - Saves the best checkpoint to a temp directory, which is reused
            by subsequent predict() calls.

        Args:
            train_df: Training DataFrame (only target_col is used).
            val_df:   Optional validation DataFrame for early stopping.

        Returns:
            self
        """
        try:
            from chronos import ChronosPipeline
            from chronos.training import ChronosConfig, ChronosDataset
        except ImportError:
            raise ImportError(
                "Fine-tuning requires: pip install amazon-chronos-t5 transformers accelerate"
            )

        # Load pretrained weights if not already loaded
        if self._pipeline is None:
            self._load_pipeline()

        # Restore the physical hourly timeline before building sliding windows.
        # The feature pipeline drops rows with missing exogenous features, leaving
        # gaps in the DatetimeIndex.  _make_sliding_windows relies on array
        # position = time step; any gap would misalign the context/forecast pairs.
        # Reindexing to a contiguous hourly frequency fills those gaps with NaN;
        # _make_sliding_windows already skips windows whose forecast target contains
        # NaN (see the `np.isnan(tgt_win).any()` guard), so no extra filtering needed.
        if isinstance(train_df.index, pd.DatetimeIndex):
            full_idx = pd.date_range(
                train_df.index.min(), train_df.index.max(), freq="h")
            train_series = train_df[self.target_col].reindex(
                full_idx).astype(np.float32)
        else:
            train_series = train_df[self.target_col].dropna().astype(
                np.float32)

        if isinstance(val_df.index, pd.DatetimeIndex) if val_df is not None else False:
            val_full_idx = pd.date_range(
                val_df.index.min(), val_df.index.max(), freq="h")
            val_series = val_df[self.target_col].reindex(
                val_full_idx).astype(np.float32)
        else:
            val_series = val_df[self.target_col].dropna().astype(
                np.float32) if val_df is not None else None

        self._logger.info(
            "Fine-tuning Chronos on '%s': %d valid training obs (of %d total after reindex), "
            "context=%d, horizon=%d …",
            self.target_col, int(np.asarray(
                train_series).size - np.isnan(np.asarray(train_series)).sum()),
            len(train_series), self.context_length, self.horizon,
        )

        cfg = self.finetune_config
        best_val_loss = float("inf")
        patience_counter = 0
        checkpoint_dir = Path(f"/tmp/chronos_ft_{self.target_col}")
        checkpoint_dir.mkdir(exist_ok=True)

        optimizer = torch.optim.AdamW(
            self._pipeline.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        train_dataset = _make_sliding_windows(
            np.asarray(train_series, dtype=np.float32),
            context_length=self.context_length,
            horizon=self.horizon,
        )
        val_dataset = (
            _make_sliding_windows(
                np.asarray(val_series, dtype=np.float32),
                context_length=self.context_length,
                horizon=self.horizon,
            )
            if val_series is not None else None
        )

        # Build cosine LR scheduler
        total_steps = cfg["num_epochs"] * \
            max(1, len(train_dataset) // cfg["batch_size"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        self._pipeline.model.train()

        for epoch in range(cfg["num_epochs"]):
            train_loss = _run_epoch(
                self._pipeline, train_dataset, optimizer, scheduler,
                batch_size=cfg["batch_size"], device=self.device, train=True,
            )

            val_loss = None
            if val_dataset is not None:
                val_loss = _run_epoch(
                    self._pipeline, val_dataset, optimizer, scheduler,
                    batch_size=cfg["batch_size"], device=self.device, train=False,
                )
                self._logger.info(
                    "  Epoch %d/%d — train_loss=%.4f  val_loss=%.4f",
                    epoch + 1, cfg["num_epochs"], train_loss, val_loss,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._pipeline.model.save_pretrained(str(checkpoint_dir))
                    self._logger.info(
                        "    ✓ New best val_loss — checkpoint saved.")
                else:
                    patience_counter += 1
                    if patience_counter >= cfg["early_stopping_patience"]:
                        self._logger.info(
                            "  Early stopping at epoch %d.", epoch + 1)
                        break
            else:
                self._logger.info(
                    "  Epoch %d/%d — train_loss=%.4f", epoch +
                    1, cfg["num_epochs"], train_loss
                )

        # Reload best checkpoint
        if val_dataset is not None and (checkpoint_dir / "config.json").exists():
            self._logger.info(
                "Reloading best checkpoint from %s …", checkpoint_dir)
            self._load_pipeline(checkpoint_dir)

        self._is_finetuned = True
        self._ft_checkpoint = checkpoint_dir
        self.is_fitted = True
        self._logger.info("Fine-tuning complete.")
        return self

    # ── predict ───────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate horizon-step probabilistic forecasts (median) for each
        row of df, using the last context_length values of target_col as
        context for each forecast origin.

        Batched: all origins are processed in a single ChronosPipeline call.

        Args:
            df: Feature DataFrame containing target_col. Must have at least
                context_length rows before each forecast origin.

        Returns:
            Array of shape (len(df), self.horizon) — median point forecasts.
        """
        if self._pipeline is None:
            # Zero-shot: load pretrained weights on first call
            self._load_pipeline()
            self.is_fitted = True

        series = df[self.target_col].values.astype(np.float32)
        n = len(series)

        # Build context tensors: for each origin i, context = series[i-ctx:i]
        # Origins that don't have enough history get left-padded with NaN
        # (Chronos handles NaN as missing — pads internally with zeros)
        ctx = self.context_length
        contexts = []
        for i in range(n):
            start = max(0, i - ctx)
            window = series[start:i]
            if len(window) < ctx:
                pad = np.full(ctx - len(window), np.nan, dtype=np.float32)
                window = np.concatenate([pad, window])
            contexts.append(torch.tensor(window, dtype=torch.float32))

        # Stack into (N, ctx) tensor
        context_tensor = torch.stack(contexts)  # (N, ctx)

        self._logger.info(
            "Running Chronos inference on %d origins (horizon=%d, ctx=%d) …",
            n, self.horizon, ctx,
        )

        # Batch inference to prevent OOM
        batch_size = 32
        forecast_list = []
        with torch.inference_mode():
            for i in range(0, len(context_tensor), batch_size):
                batch = context_tensor[i: i + batch_size]
                f = self._pipeline.predict(
                    batch,
                    prediction_length=self.horizon,
                    num_samples=self.num_samples,
                    limit_prediction_length=False,
                )
                forecast_list.append(f.numpy())

        # forecast shape: (N, num_samples, horizon) — take median across samples
        forecast_np = np.concatenate(forecast_list, axis=0)
        median = np.median(forecast_np, axis=1)  # (N, H)

        self._logger.info("Inference complete.")
        return median.astype(np.float32)

    def predict_quantiles(
        self, df: pd.DataFrame
    ) -> dict[float, np.ndarray]:
        """
        Return per-quantile forecast arrays (N, H) for uncertainty intervals.

        Returns:
            Dict mapping quantile level → array of shape (N, horizon).
            e.g. {0.1: lower_bound, 0.5: median, 0.9: upper_bound}
        """
        if self._pipeline is None:
            self._load_pipeline()
            self.is_fitted = True

        series = df[self.target_col].values.astype(np.float32)
        n, ctx = len(series), self.context_length
        contexts = []
        for i in range(n):
            start = max(0, i - ctx)
            window = series[start:i]
            if len(window) < ctx:
                pad = np.full(ctx - len(window), np.nan, dtype=np.float32)
                window = np.concatenate([pad, window])
            contexts.append(torch.tensor(window, dtype=torch.float32))

        context_tensor = torch.stack(contexts)
        # Batch inference for quantiles (smaller batch size due to more samples)
        batch_size = 16
        forecast_list = []
        num_s = max(self.num_samples, 100)

        with torch.inference_mode():
            for i in range(0, len(context_tensor), batch_size):
                batch = context_tensor[i: i + batch_size]
                f = self._pipeline.predict(
                    batch,
                    prediction_length=self.horizon,
                    num_samples=num_s,
                    limit_prediction_length=False,
                )
                forecast_list.append(f.numpy())

        forecast_np = np.concatenate(forecast_list, axis=0)   # (N, samples, H)
        return {
            q: np.quantile(forecast_np, q, axis=1).astype(np.float32)
            for q in self.quantile_levels
        }

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save the model. Fine-tuned weights go into path/weights/,
        config and metadata into path/meta.pkl.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "target_col":      self.target_col,
            "horizon":         self.horizon,
            "model_id":        self.model_id,
            "context_length":  self.context_length,
            "num_samples":     self.num_samples,
            "finetune_config": self.finetune_config,
            "quantile_levels": self.quantile_levels,
            "is_finetuned":    self._is_finetuned,
            "device":          self.device,
        }
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        if self._is_finetuned and self._pipeline is not None:
            weights_dir = path / "weights"
            self._pipeline.model.save_pretrained(str(weights_dir))
            self._logger.info("Fine-tuned weights saved → %s/weights/", path)

        self._logger.info("ChronosForecaster saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ChronosForecaster":
        """Restore a saved ChronosForecaster from disk."""
        path = Path(path)
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        instance = cls(
            target_col=meta["target_col"],
            horizon=meta["horizon"],
            model_id=meta["model_id"],
            context_length=meta["context_length"],
            num_samples=meta["num_samples"],
            finetune_config=meta["finetune_config"],
            quantile_levels=meta["quantile_levels"],
            device=meta["device"],
        )
        instance._is_finetuned = meta["is_finetuned"]

        weights_dir = path / "weights"
        if instance._is_finetuned and weights_dir.exists():
            instance._load_pipeline(weights_dir)
        else:
            instance._load_pipeline()

        instance.is_fitted = True
        return instance


# ── Training helpers ──────────────────────────────────────────────────────────

def _auto_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_sliding_windows(
    series: np.ndarray,
    context_length: int,
    horizon: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build (context, target) pairs by sliding a window over the series.

    Returns a list of (context_window, forecast_window) tuples,
    where context_window has shape (context_length,) and
    forecast_window has shape (horizon,).
    """
    windows = []
    total = context_length + horizon
    for i in range(len(series) - total + 1):
        ctx_win = series[i: i + context_length]
        tgt_win = series[i + context_length: i + total]
        if not (np.isnan(ctx_win).all() or np.isnan(tgt_win).any()):
            windows.append((ctx_win, tgt_win))
    return windows


def _run_epoch(
    pipeline,
    dataset: list[tuple[np.ndarray, np.ndarray]],
    optimizer,
    scheduler,
    batch_size: int,
    device: str,
    train: bool,
) -> float:
    """
    Run one training or validation epoch over the dataset.

    Uses teacher-forcing: the model receives the context window and is
    trained to predict the forecast window via cross-entropy on tokenised
    values (Chronos's internal training objective).

    Returns:
        Mean loss over all batches.
    """
    import random
    pipeline.model.train(train)
    indices = list(range(len(dataset)))
    if train:
        random.shuffle(indices)

    total_loss = 0.0
    n_batches = 0

    for batch_start in range(0, len(indices), batch_size):
        batch_idx = indices[batch_start: batch_start + batch_size]
        ctx_batch = torch.tensor(
            np.stack([dataset[i][0] for i in batch_idx]), dtype=torch.float32
        ).to(device)
        tgt_batch = torch.tensor(
            np.stack([dataset[i][1] for i in batch_idx]), dtype=torch.float32
        ).to(device)

        with torch.set_grad_enabled(train):
            # Chronos training: pass context + labels to the model
            # The pipeline's model is a ChronosModel which wraps T5ForConditionalGeneration
            outputs = pipeline.model(
                context=ctx_batch,
                target=tgt_batch,
            )
            loss = outputs.loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
