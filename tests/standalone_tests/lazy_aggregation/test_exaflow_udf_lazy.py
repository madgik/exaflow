import numpy as np

from exaflow.algorithms.exareme3.utils.registry import exareme3_registry
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.exareme3.utils.registry import get_udf_registry_key
from exaflow.worker.exareme3.lazy_aggregation import RecordingAggClient
from exaflow.worker.exareme3.lazy_aggregation import lazy_agg


def _wrap_udf_for_worker(func):
    key = get_udf_registry_key(func)
    if exareme3_registry.lazy_aggregation_enabled(key):
        return lazy_agg(func)
    return func


def test_exareme3_udf_enables_lazy_by_default():
    calls = RecordingAggClient()

    @exareme3_udf(with_aggregation_server=True)
    def udf(agg_client):
        a = agg_client.sum(np.array([1.0], dtype=float))
        b = agg_client.sum(np.array([2.0], dtype=float))
        return float(np.asarray(a, dtype=float)[0] + np.asarray(b, dtype=float)[0])

    total = _wrap_udf_for_worker(udf)(calls)
    assert total == 3.0
    assert calls.calls == [("batch", 2)]


def test_exareme3_udf_lazy_can_be_disabled():
    calls = RecordingAggClient()

    @exareme3_udf(with_aggregation_server=True, enable_lazy_aggregation=False)
    def udf(agg_client):
        a = agg_client.sum(np.array([1.0], dtype=float))
        b = agg_client.sum(np.array([2.0], dtype=float))
        return float(np.asarray(a, dtype=float)[0] + np.asarray(b, dtype=float)[0])

    total = _wrap_udf_for_worker(udf)(calls)
    assert total == 3.0
    assert calls.calls == [("sum", 1), ("sum", 1)]
