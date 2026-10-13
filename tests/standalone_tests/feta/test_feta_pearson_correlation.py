import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.statistics.pearson_correlation import (
    _format_result_matrices,
)
from exaflow.algorithms.federated.statistics.pearson_correlation import (
    FederatedPearsonCorrelation,
)
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta.feta_pearson_correlation import FetaPearsonCorrelation
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation


class IdentityAggregation:
    def sum(self, value):
        return value


def data(n=39):
    rng = np.random.default_rng(19)
    return pd.DataFrame(rng.normal(size=(n, 5)), columns=["x0", "x1", "x2", "y0", "y1"])


def inputs(frame):
    return Inputdata(data_model="test:1.0", datasets=["dataset"], variables=list(frame))


@pytest.mark.parametrize("sizes", [[39], [17, 13, 9], [0, 20, 19]])
@pytest.mark.parametrize("omitted_x", [False, True])
def test_pearson_parity_with_mocked_aggregation(sizes, omitted_x):
    frame = data(sum(sizes))
    boundaries = np.cumsum([0, *sizes])
    parts = [
        frame.iloc[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    runtime, workers, aggregation = federation(parts)
    x_names = [] if omitted_x else ["x2", "x0", "x1"]
    y_names = ["y1", "y0"]
    model = FetaPearsonCorrelation(
        parameters={"alpha": 0.05}, x_names=x_names, y_names=y_names
    )
    with runtime:
        columns = {"y": y_names}
        if x_names:
            columns["x"] = x_names
        tensors = runtime.initialize(inputs(frame), columns)
        result = model.run(**tensors)
        assert result.n_obs == len(frame)
        assert runtime.step == 6
        for step in range(1, 7):
            events = [
                kind for kind, event_step in aggregation.events if event_step == step
            ]
            assert events == ["prepare", *("submit" for _ in workers), "fetch"]
    reference = FederatedPearsonCorrelation(IdentityAggregation()).corrcoef(
        data=frame, x_vars=x_names or y_names, y_vars=y_names, alpha=0.05
    )
    matrices = _format_result_matrices(
        vars(reference), row_names=x_names or y_names, column_names=y_names
    )
    for field, expected in zip(
        ("correlations", "p_values", "ci_hi", "ci_lo"), matrices
    ):
        actual = getattr(result, field)
        assert actual["variables"] == expected["variables"]
        for column in y_names:
            np.testing.assert_allclose(
                actual[column], expected[column], rtol=1e-10, atol=1e-12, equal_nan=True
            )
    assert all(not worker.service.store.contexts for worker in workers)
    assert not aggregation.rounds


def test_local_reductions_broadcasting_and_matmul_orientation():
    frame = data()
    runtime, workers, aggregation = federation([frame.iloc[:17], frame.iloc[17:]])
    with runtime:
        tensors = runtime.initialize(
            inputs(frame), {"x": ["x0", "x1", "x2"], "y": ["y0", "y1"]}
        )
        x = tensors["x"]
        y = tensors["y"]
        local_sums = x.sum(axis=1)
        assert isinstance(local_sums, FedTensor)
        assert runtime.step == 0
        np.testing.assert_allclose(
            local_sums.sum(axis=0), frame[["x0", "x1", "x2"]].to_numpy().sum()
        )
        assert len(x.T) == 3
        means = x.sum(axis=0) / len(x)
        centered = x - means
        np.testing.assert_allclose(centered.sum(axis=0), np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(
            (2 + x * 3 - np.array([1.0, 2.0, 3.0])).sum(axis=0),
            (2 + frame[["x0", "x1", "x2"]].to_numpy() * 3 - [1, 2, 3]).sum(axis=0),
        )
        np.testing.assert_allclose(
            (np.array([1.0, 2.0, 3.0]) - x).sum(axis=0),
            ([1, 2, 3] - frame[["x0", "x1", "x2"]].to_numpy()).sum(axis=0),
        )
        product = y.T @ x
        assert product.shape == (2, 3)
        np.testing.assert_allclose(
            product,
            frame[["y0", "y1"]].to_numpy().T @ frame[["x0", "x1", "x2"]].to_numpy(),
        )
        with pytest.raises(ValueError, match="partitioning"):
            x @ x.T
        with pytest.raises(TypeError, match="converted"):
            np.asarray(x)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 10])
def test_observation_edges_and_constant_features(n):
    frame = pd.DataFrame({"x": np.ones(n), "y": np.arange(n, dtype=float)})
    runtime, workers, _ = federation([frame])
    with runtime:
        tensors = runtime.initialize(inputs(frame), {"x": ["x"], "y": ["y"]})
        x = tensors["x"]
        y = tensors["y"]
        model = FetaPearsonCorrelation(
            parameters={"alpha": 0.05}, x_names=["x"], y_names=["y"]
        )
        if n <= 2:
            with pytest.raises(
                BadUserInput, match="empty data|Not enough observations"
            ):
                model.run(x=x, y=y)
        else:
            result = model.run(x=x, y=y)
            assert result.correlations["y"] == [0.0]
    assert not workers[0].service.store.contexts


def test_skip_insufficient_workers_before_first_aggregation():
    frame = data(27)
    runtime, workers, aggregation = federation(
        [frame.iloc[:7], frame.iloc[7:]], minimum_rows=10
    )
    with runtime:
        tensors = runtime.initialize(inputs(frame), {"x": ["x0"], "y": ["y0"]})
        x = tensors["x"]
        y = tensors["y"]
        assert len(x) == 20
        assert set(runtime.workers) == {"worker-1"}
        assert not workers[0].service.store.contexts
    runtime, workers, _ = federation([frame.iloc[:7]], minimum_rows=10)
    with pytest.raises(Exception, match="No workers had sufficient"):
        with runtime:
            runtime.initialize(inputs(frame), {"x": ["x0"], "y": ["y0"]})


def test_missing_values_fail_and_clean_all_workers():
    frame = data()
    frame.loc[0, "x0"] = np.nan
    runtime, workers, _ = federation([frame.iloc[:20], frame.iloc[20:]])
    with pytest.raises(ValueError, match="missing input values"):
        with runtime:
            runtime.initialize(inputs(frame), {"x": ["x0"], "y": ["y0"]})
    assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize(
    "expression, reference",
    [
        (lambda x: (x / 3) ** 2, lambda a: (a / 3) ** 2),
        (lambda x: 2 / (x * x + 1), lambda a: 2 / (a * a + 1)),
        (lambda x: 2.0**x, lambda a: 2.0**a),
        (lambda x: np.float64(2) + x, lambda a: 2 + a),
    ],
)
def test_arithmetic_operators_against_numpy_across_uneven_workers(
    expression, reference
):
    frame = data()
    columns = ["x0", "x1", "x2"]
    runtime, workers, _ = federation(
        [frame.iloc[:7], frame.iloc[7:20], frame.iloc[20:]]
    )
    with runtime:
        tensors = runtime.initialize(inputs(frame), {"x": columns, "y": ["y0"]})
        x = tensors["x"]
        _ = tensors["y"]
        np.testing.assert_allclose(
            expression(x).sum(axis=0), reference(frame[columns].to_numpy()).sum(axis=0)
        )
    assert all(not worker.service.store.contexts for worker in workers)


def test_literal_matrix_products_and_transposed_reductions():
    frame = data()
    columns = ["x0", "x1", "x2"]
    array = frame[columns].to_numpy()
    runtime, _, _ = federation([frame.iloc[:17], frame.iloc[17:]])
    with runtime:
        tensors = runtime.initialize(inputs(frame), {"x": columns, "y": ["y0"]})
        x = tensors["x"]
        _ = tensors["y"]
        right = np.arange(6.0, dtype=float).reshape(3, 2)
        left = np.arange(6.0, dtype=float).reshape(2, 3)
        product = x @ right
        assert isinstance(product, FedTensor)
        np.testing.assert_allclose(product.sum(axis=0), (array @ right).sum(axis=0))
        product = left @ x.T
        assert isinstance(product, FedTensor)
        np.testing.assert_allclose(product.sum(axis=1), (left @ array.T).sum(axis=1))
        np.testing.assert_allclose(x.T.sum(axis=1), array.sum(axis=0))
        np.testing.assert_allclose(x.sum(), array.sum())


def test_algorithm_inputs_implement_public_typed_tensor_interface():
    from inspect import signature

    from exaflow.algorithms.feta.feta_pearson_correlation import FetaPearsonCorrelation
    from exaflow.feta.tensor import RuntimeFedTensor

    assert issubclass(RuntimeFedTensor, FedTensor)
    assert (
        signature(FetaPearsonCorrelation.run).parameters["x"].annotation
        == FedTensor | None
    )
    assert signature(FetaPearsonCorrelation.run).parameters["y"].annotation is FedTensor
    runtime, _, _ = federation([data(20)])
    with runtime:
        tensors = runtime.initialize(inputs(data(20)), {"x": ["x0"], "y": ["y0"]})
        x = tensors["x"]
        y = tensors["y"]
        assert isinstance(x, FedTensor) and isinstance(y, FedTensor)
        assert isinstance(x.T, FedTensor)
        assert x.shape == (20, 1) and x.T.shape == (1, 20)
        assert x.dtype == np.dtype("float64")
        assert x.partition_axis == 0 and x.T.partition_axis == 1
