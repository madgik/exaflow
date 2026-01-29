from __future__ import annotations

import numpy as np
import pyarrow as pa


def _to_numpy(x) -> np.ndarray:
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
