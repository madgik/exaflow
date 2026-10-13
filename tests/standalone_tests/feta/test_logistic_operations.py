import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.feta.inputdata import Inputdata
from exaflow.feta.operations import LocalOperation
from exaflow.feta.operations import LocalTensorRef
from exaflow.feta.operations import OperationType
from exaflow.feta.operations import TensorMetadata
from exaflow.feta.operations import infer_operation
from tests.standalone_tests.feta.helpers import federation


def test_exp_log_where_preserve_numpy_values_and_rows():
    data = pd.DataFrame({"v": [0.5, 1.0, 2.0, 3.0]})
    runtime, _, _ = federation([data.iloc[:0], data.iloc[:2], data.iloc[2:]])
    with runtime:
        y = runtime.initialize(
            Inputdata(data_model="test:1.0", datasets=["a"], variables=["v"]),
            {"y": ["v"]},
        )["y"]
        np.testing.assert_allclose(fnp.sum(fnp.exp(y)), np.exp(data.v).sum())
        np.testing.assert_allclose(fnp.sum(fnp.log(y)), np.log(data.v).sum())
        result = fnp.where(y == 1.0, y * 2.0, 0.0)
        assert result.shape == y.shape
        assert fnp.sum(result) == 2.0
        with pytest.raises(ValueError, match="boolean"):
            fnp.where(y, y, 0.0)
        with pytest.raises(ValueError, match="unaligned"):
            fnp.where(y == 1.0, y[y[:, 0] == 1.0], 0.0)
    for function in (fnp.exp, fnp.log):
        np.testing.assert_array_equal(
            function(data.v.to_numpy()),
            getattr(np, function.__name__)(data.v.to_numpy()),
        )
    np.testing.assert_array_equal(
        fnp.where(np.array([True, False]), 1.0, 2.0), [1.0, 2.0]
    )


def test_where_rejects_text_output():
    operation = LocalOperation(
        OperationType.WHERE, (LocalTensorRef("mask"), "label", "other")
    )
    mask = TensorMetadata((3, 1), "bool", 0, "ctx")
    label = TensorMetadata((), "U5", -1, "")
    with pytest.raises(ValueError, match="numerical"):
        infer_operation(operation, [mask, label, label], 1024)
