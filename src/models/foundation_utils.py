"""Shared context batching for target-only foundation forecasts."""

from collections.abc import Iterator

import numpy as np


def iter_context_batches(
    series: np.ndarray,
    context_length: int,
    batch_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield full contexts ending at ``origin - 1`` in origin order."""
    values = np.asarray(series, dtype=np.float32)
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    for start in range(context_length, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        contexts = np.stack(
            [values[origin - context_length:origin] for origin in range(start, stop)]
        )
        yield start, contexts
