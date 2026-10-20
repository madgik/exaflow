from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa


class BadInputError(ValueError):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def as_numeric_array(data, *, input_name: str, dtype=float) -> np.ndarray:
    """Preserve existing NumPy numeric conversions with safe input errors."""
    try:
        if isinstance(data, pd.Series) and isinstance(data.dtype, np.dtype):
            return data.to_numpy(dtype=dtype, copy=False)
        return np.asarray(data, dtype=dtype)
    except (TypeError, ValueError, OverflowError):
        raise BadInputError(f"{input_name} must contain numeric values.") from None


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
    return as_numeric_array(x, input_name="Input")
