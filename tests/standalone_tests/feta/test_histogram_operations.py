import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.feta.inputdata import Inputdata
from tests.standalone_tests.feta.helpers import federation


def tensor(runtime):
    return runtime.initialize(
        Inputdata(data_model="test:1.0", datasets=["a"], variables=["v"]), {"y": ["v"]}
    )["y"][:, 0]


@pytest.mark.parametrize("values", [[-1, 0, 0.5, 1, 2, 2], [1] * 6])
@pytest.mark.parametrize("bins", [3, np.array([-1.0, 0.0, 1.0, 2.0])])
def test_histogram_numpy_parity(values, bins):
    runtime, workers, aggregation = federation(
        [pd.DataFrame({"v": values[:2]}), pd.DataFrame({"v": values[2:]})]
    )
    with runtime:
        counts, edges = fnp.histogram(tensor(runtime), bins=bins)
        expected_counts, expected_edges = np.histogram(values, bins=bins)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(edges, expected_edges)
        assert counts.dtype == np.dtype("int64")
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize(
    "edges",
    [
        np.array([1.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, np.inf]),
    ],
)
def test_invalid_edges_rejected_before_worker_dispatch(edges):
    runtime, workers, _ = federation([pd.DataFrame({"v": [1, 2, 3]})])
    with runtime:
        v = tensor(runtime)
        before = len(workers[0].operations)
        with pytest.raises(ValueError, match="strictly increasing"):
            fnp.histogram(v, bins=edges)
        assert len(workers[0].operations) == before


def test_local_numpy_histogram():
    data = np.arange(10.0)
    counts, edges = fnp.histogram(data, bins=3, range=(1, 7))
    expected = np.histogram(data, bins=3, range=(1, 7))
    np.testing.assert_array_equal(counts, expected[0])
    np.testing.assert_array_equal(edges, expected[1])


def test_filtered_empty_partitions_and_explicit_range():
    parts = [pd.DataFrame({"v": [-3.0, -2.0]}), pd.DataFrame({"v": [0.0, 1.0, 2.0]})]
    runtime, _, _ = federation(parts)
    with runtime:
        v = tensor(runtime)
        sample = v[v == 1.0]
        counts, edges = fnp.histogram(sample, bins=2, range=(0, 2))
        np.testing.assert_array_equal(counts, [0, 1])
        np.testing.assert_array_equal(edges, [0, 1, 2])


def test_worker_rejects_invalid_histogram_edges():
    from exaflow.feta.operations import LocalOperation
    from exaflow.feta.operations import LocalTensorRef
    from exaflow.feta.operations import OperationType
    from exaflow.protos.feta import feta_pb2 as pb

    runtime, workers, _ = federation([pd.DataFrame({"v": [1, 2, 3]})])
    with runtime:
        v = tensor(runtime)
        descriptor = v.descriptors[workers[0].worker_id]
        request = pb.ExecuteRequest(
            context_id=runtime.context_id,
            aggregation_step=1,
            operation=LocalOperation(
                OperationType.HISTOGRAM,
                (LocalTensorRef(descriptor.tensor_id), np.array([1.0, 0.0])),
            ).to_proto(),
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            workers[0].service.execute(request)
