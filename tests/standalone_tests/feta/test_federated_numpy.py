import numpy as np
import pytest

from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_pearson_correlation import data
from tests.standalone_tests.feta.test_feta_pearson_correlation import inputs


@pytest.mark.parametrize("function_syntax", [False, True])
def test_reductions_and_transpose_preserve_federation_semantics(function_syntax):
    frame = data(39)
    runtime, workers, aggregation = federation([frame.iloc[:17], frame.iloc[17:]])
    reduce = fnp.sum if function_syntax else lambda a, axis=None: a.sum(axis=axis)
    transpose = fnp.transpose if function_syntax else lambda a: a.T
    columns = ["x2", "x0", "x1"]
    array = frame[columns].to_numpy()

    with runtime:
        tensors = runtime.initialize(inputs(frame), {"x": columns, "y": ["y1", "y0"]})
        x = tensors["x"]
        y = tensors["y"]
        yt = transpose(y)
        assert isinstance(yt, FedTensor)
        assert yt.partition_axis == 1 and yt.shape == (2, 39)
        row_sums = reduce(x, axis=1)
        assert isinstance(row_sums, FedTensor)
        assert row_sums.shape == (39,) and runtime.step == 0

        feature_sums = reduce(x, axis=0)
        assert isinstance(feature_sums, np.ndarray)
        np.testing.assert_allclose(feature_sums, array.sum(axis=0))
        np.testing.assert_allclose(reduce(row_sums), array.sum())
        full_sum = reduce(x, axis=None)
        assert isinstance(full_sum, float)
        np.testing.assert_allclose(full_sum, array.sum())
        product = yt @ x
        np.testing.assert_allclose(product, frame[["y1", "y0"]].to_numpy().T @ array)
        assert runtime.step == 4
        assert sum(kind == "prepare" for kind, _ in aggregation.events) == 4

        with pytest.raises(ValueError, match="axis"):
            reduce(x, axis=2)
        assert runtime.step == 4

    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_concrete_arrays_use_numpy_results(dtype):
    array = np.arange(12, dtype=dtype).reshape(4, 3)
    for axis in (None, 0, 1):
        actual = fnp.sum(array, axis=axis)
        expected = np.sum(array, axis=axis)
        np.testing.assert_array_equal(actual, expected)
        assert type(actual) is type(expected)
        assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(fnp.transpose(array), np.transpose(array))


@pytest.mark.parametrize("value", [[1, 2], np.array(["a"]), np.array([True])])
def test_unsupported_inputs_are_rejected(value):
    for function in (fnp.sum, fnp.transpose):
        with pytest.raises(TypeError, match="numerical NumPy array"):
            function(value)


@pytest.mark.parametrize(
    "kwargs", [{"keepdims": True}, {"dtype": float}, {"out": None}]
)
def test_unsupported_sum_parameters_are_rejected(kwargs):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        fnp.sum(np.ones((2, 3)), **kwargs)


def test_custom_transpose_axes_are_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        fnp.transpose(np.ones((2, 3)), axes=(1, 0))
