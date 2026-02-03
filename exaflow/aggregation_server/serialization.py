import json
from typing import Sequence

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray


def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes using Apache Arrow."""
    tensor = pa.Tensor.from_numpy(ndarray)
    sink = pa.BufferOutputStream()
    pa.ipc.write_tensor(tensor, sink)
    return sink.getvalue().to_pybytes()


def bytes_to_ndarray(tensor_bytes: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes using Apache Arrow."""
    reader = pa.BufferReader(tensor_bytes)
    tensor = pa.ipc.read_tensor(reader)
    return tensor.to_numpy()


def values_to_bytes(values: Sequence[object]) -> bytes:
    """Serialize arbitrary values to bytes via JSON."""

    def _jsonify(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return value

    payload = [_jsonify(value) for value in values]
    return json.dumps(payload).encode("utf-8")


def bytes_to_values(payload: bytes) -> NDArray:
    """Deserialize arbitrary values from JSON bytes."""
    return np.asarray(json.loads(payload.decode("utf-8")), dtype=object)
