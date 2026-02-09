import numpy as np

from exaflow.worker.exareme3.lazy_aggregation import RecordingAggClient
from exaflow.worker.exareme3.lazy_aggregation import lazy_agg


@lazy_agg()
def outer_with_inner(agg_client):
    @lazy_agg()
    def inner(client):
        x = client.sum(np.array([1.0], dtype=float))
        y = client.sum(np.array([2.0], dtype=float))
        return float(np.asarray(x, dtype=float)[0] + np.asarray(y, dtype=float)[0])

    a = agg_client.sum(np.array([3.0], dtype=float))
    b = inner(agg_client)
    return float(np.asarray(a, dtype=float)[0]), b


def test_nested_functions_are_rewritten_and_batched():
    agg = RecordingAggClient()
    a, b = outer_with_inner(agg)

    assert a == 3.0
    assert b == 3.0
    # Inner batches two sums and outer single sum; order may reorder but both must appear.
    assert agg.calls == [("batch", 2), ("batch", 1)]
