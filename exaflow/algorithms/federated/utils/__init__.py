from __future__ import annotations

import numpy as np
import pyarrow as pa


class BadInputError(ValueError):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def to_numpy(x) -> np.ndarray:
    """Convert input (Arrow Table/Array or list/array) to NumPy array."""
    if isinstance(x, pa.Table):
        # Prefer zero-copy Arrow->NumPy via pandas destruction when possible
        try:
            return x.to_pandas(split_blocks=True, self_destruct=True).to_numpy(
                dtype=float
            )
        except Exception:
            return x.to_pandas().to_numpy(dtype=float)
    if isinstance(x, (pa.Array, pa.ChunkedArray)):
        try:
            return x.to_numpy(zero_copy_only=True)
        except Exception:
            return x.to_numpy(zero_copy_only=False)
    return np.asarray(x, dtype=float)


def fused_sum(agg_client, *arrays) -> list[np.ndarray]:
    """Sum several arrays across workers in a single aggregation round-trip.

    Each array is flattened, concatenated, summed once through ``agg_client``
    and split back into float arrays with the original shapes.
    """
    arrays = [np.asarray(array, dtype=float) for array in arrays]
    flat = np.concatenate([array.reshape(-1) for array in arrays])
    total = np.asarray(agg_client.sum(flat), dtype=float).reshape(-1)
    offsets = np.cumsum([array.size for array in arrays])[:-1]
    return [
        part.reshape(array.shape)
        for part, array in zip(np.split(total, offsets), arrays)
    ]


def fused_min_max(agg_client, mins, maxs) -> tuple[np.ndarray, np.ndarray]:
    """Global elementwise min of ``mins`` and max of ``maxs`` in one round-trip.

    Uses a single MIN aggregation since ``max(x) == -min(-x)`` exactly in IEEE
    arithmetic (NaN and inf propagate the same way).
    """
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    flat = np.concatenate([mins.reshape(-1), -maxs.reshape(-1)])
    total = np.asarray(agg_client.min(flat), dtype=float).reshape(-1)
    return (
        total[: mins.size].reshape(mins.shape),
        -total[mins.size :].reshape(maxs.shape),
    )
