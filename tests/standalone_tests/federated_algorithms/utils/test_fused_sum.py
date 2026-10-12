import numpy as np

from exaflow.algorithms.federated.utils import fused_min_max
from exaflow.algorithms.federated.utils import fused_sum


class _CountingSumClient:
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.calls = 0

    def sum(self, values):
        self.calls += 1
        return np.asarray(values, dtype=float) * self.n_workers


def test_fused_sum_single_round_trip_preserves_shapes():
    client = _CountingSumClient(n_workers=3)
    matrix = np.arange(6, dtype=float).reshape(2, 3)
    vector = np.array([1.0, 2.0])
    empty = np.zeros((0, 0))

    summed_matrix, summed_vector, summed_empty, scalars = fused_sum(
        client, matrix, vector, empty, [4.0, 5.0]
    )

    assert client.calls == 1
    np.testing.assert_array_equal(summed_matrix, matrix * 3)
    np.testing.assert_array_equal(summed_vector, vector * 3)
    assert summed_empty.shape == (0, 0)
    np.testing.assert_array_equal(scalars, [12.0, 15.0])


class _TwoWorkerMinClient:
    def __init__(self, other):
        self.other = np.asarray(other, dtype=float)
        self.calls = 0

    def min(self, values):
        self.calls += 1
        return np.minimum(np.asarray(values, dtype=float), self.other)


def test_fused_min_max_single_round_trip():
    local_min = np.array([1.0, np.inf])
    local_max = np.array([5.0, -np.inf])
    other_min = np.array([0.0, 2.0])
    other_max = np.array([7.0, 3.0])
    client = _TwoWorkerMinClient(np.concatenate([other_min, -other_max]))

    global_min, global_max = fused_min_max(client, local_min, local_max)

    assert client.calls == 1
    np.testing.assert_array_equal(global_min, [0.0, 2.0])
    np.testing.assert_array_equal(global_max, [7.0, 3.0])
