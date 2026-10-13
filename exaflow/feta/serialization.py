"""FETA-owned Arrow array transport."""

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
