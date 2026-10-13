import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.feta.inputdata import Inputdata
from exaflow.feta.lifecycle import ResourceLimitError
from exaflow.feta.operations import LocalOperation
from exaflow.feta.operations import LocalTensorRef
from exaflow.feta.operations import OperationType
from exaflow.feta.operations import TensorMetadata
from exaflow.feta.operations import infer_operation
from exaflow.protos.feta import feta_pb2 as pb
from tests.standalone_tests.feta.helpers import federation


def test_stack_vectors_matrices_and_ones_across_empty_partitions():
    frame = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [2.0, 4.0, 6.0]})
    runtime, _, _ = federation([frame.iloc[:0], frame.iloc[:1], frame.iloc[1:]])
    with runtime:
        y = runtime.initialize(
            Inputdata(data_model="test:1.0", datasets=["a"], variables=list(frame)),
            {"y": list(frame)},
        )["y"]
        ones = fnp.ones_like(y)
        assert isinstance(ones, FedTensor)
        assert ones.shape == y.shape
        assert ones.dtype == y.dtype
        assert fnp.sum(ones) == 6
        design = fnp.column_stack([ones, y[:, 1]])
        expected = np.column_stack([np.ones_like(frame.to_numpy()), frame.b])
        np.testing.assert_array_equal(fnp.sum(design, axis=0), expected.sum(axis=0))
        np.testing.assert_array_equal(design.T @ design, expected.T @ expected)
        with pytest.raises(ValueError, match="row-partitioned"):
            fnp.column_stack([y.T])
        with pytest.raises(TypeError, match="mix"):
            fnp.column_stack([y, np.ones((3, 1))])
        with pytest.raises(ValueError, match="unaligned"):
            fnp.column_stack([y, y[y[:, 0] == 1.0]])


def test_design_allocation_limit_and_empty_request():
    op = LocalOperation(
        OperationType.COLUMN_STACK, (LocalTensorRef("a"), LocalTensorRef("a"))
    )
    metadata = TensorMetadata((3, 2), "float64", 0, "ctx")
    with pytest.raises(ResourceLimitError):
        infer_operation(op, [metadata, metadata], 64)
    with pytest.raises(ValueError, match="inputs"):
        LocalOperation.from_proto(pb.LocalOperation(type=pb.COLUMN_STACK), 1024)
    with pytest.raises(ValueError):
        fnp.column_stack([])
    text = TensorMetadata((3,), "U10", 0, "ctx")
    with pytest.raises(ValueError, match="mix text"):
        infer_operation(op, [metadata, text], 1024)


def test_numpy_design_operations_are_local():
    a, b = np.arange(3), np.arange(6).reshape(3, 2)
    np.testing.assert_array_equal(fnp.ones_like(b), np.ones_like(b))
    np.testing.assert_array_equal(fnp.column_stack([a, b]), np.column_stack([a, b]))
