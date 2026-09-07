"""Shared context construction and fingerprinting for foundation forecasts."""

from collections.abc import Iterator
import hashlib
import json

import numpy as np


def build_context_actuals(
    history: np.ndarray,
    evaluation: np.ndarray,
    context_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    '''Build canonical target-only contexts and horizon-aligned actuals.'''
    history_values = np.asarray(history, dtype=np.float32)
    evaluation_values = np.asarray(evaluation, dtype=np.float32)
    if len(history_values) < context_length:
        raise ValueError('Insufficient history for the requested context length.')
    combined = np.concatenate([history_values[-context_length:], evaluation_values])
    contexts = np.stack(
        [combined[start:start + context_length] for start in range(len(evaluation_values))]
    ).astype(np.float32)
    actuals = np.full(
        (len(evaluation_values), horizon), np.nan, dtype=np.float32
    )
    for step in range(horizon):
        available = len(evaluation_values) - step
        if available <= 0:
            break
        actuals[:available, step] = evaluation_values[step:]
    return contexts, actuals


def _update_fingerprint_array(digest, name, values, dtype):
    array = np.asarray(values, dtype=dtype)
    if np.issubdtype(array.dtype, np.floating) and np.isnan(array).any():
        array = array.copy()
        array[np.isnan(array)] = np.nan
    array = np.ascontiguousarray(array)
    descriptor = {
        'name': name,
        'dtype': array.dtype.str,
        'shape': list(array.shape),
    }
    digest.update(
        json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode('utf-8')
    )
    digest.update(array.view(np.uint8))


def foundation_input_fingerprint(
    *,
    target: str,
    origins_ns: np.ndarray,
    contexts: np.ndarray,
    actuals: np.ndarray,
    benchmark_set: str,
    run_mode: str,
    context_length: int,
    horizon: int,
) -> str:
    '''Hash the exact ordered data and metadata behind one target benchmark.'''
    metadata = {
        'target': target,
        'benchmark_set': benchmark_set,
        'run_mode': run_mode,
        'context_length': int(context_length),
        'horizon': int(horizon),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8')
    )
    _update_fingerprint_array(digest, 'origins_ns', origins_ns, '<i8')
    _update_fingerprint_array(digest, 'contexts', contexts, '<f4')
    _update_fingerprint_array(digest, 'actuals', actuals, '<f4')
    return digest.hexdigest()


def legacy_array_sha256(*arrays: np.ndarray) -> str:
    '''Recompute the array-only digest used by pre-fingerprint handoffs.'''
    digest = hashlib.sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


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
