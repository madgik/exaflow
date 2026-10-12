import numpy as np

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
