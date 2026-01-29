import inspect
import traceback

import numpy as np
import pytest

from exaflow.algorithms.exareme3.lazy_aggregation import lazy_agg
from exaflow.algorithms.exareme3.library.stats import stats


class RecordingAggClient:
    def __init__(self):
        self.calls = []

    def aggregate_batch(self, ops):
        self.calls.append(("batch", len(ops)))
        return [value for _, value in ops]

    def sum(self, value):
        self.calls.append(("sum", np.asarray(value).size))
        return value

    def min(self, value):
        self.calls.append(("min", np.asarray(value).size))
        return value

    def max(self, value):
        self.calls.append(("max", np.asarray(value).size))
        return value


GLOBAL_BUFFER = []
GLOBAL_INPUT = np.array([7.0, 8.0], dtype=float)
SIDE_EFFECT_LOG = []


def _assert_calls(actual, expected, tolerant=False):
    if tolerant:
        # Allow a sequence of sums followed by the expected batches
        assert actual[-len(expected) :] == expected
        return
    assert actual == expected


def _expect_exact(expected):
    def checker(calls):
        assert calls == expected

    return checker


@lazy_agg()
def aggregation(agg_client):
    a = 5.0

    # Global: x (independent)
    x = agg_client.sum(np.array([10.0], dtype=float))

    # Local: z derived from x
    z = np.asarray(x, dtype=float) + 1.0

    # Global: y depends on z
    y = agg_client.sum(z)

    # Local: uses a (independent)
    local_a = y[0] + 2.0

    # Global: m is independent of y/local_a, should batch with y
    m = agg_client.sum(np.array([3.0], dtype=float))

    return {
        "x_global": float(np.asarray(x, dtype=float).reshape(-1)[0]),
        "z_local": float(z.reshape(-1)[0]),
        "y_global": float(np.asarray(y, dtype=float).reshape(-1)[0]),
        "m_global": float(np.asarray(m, dtype=float).reshape(-1)[0]),
        "local_a": float(local_a),
    }


def test_dummy_aggregation_batches_and_results():
    agg = RecordingAggClient()
    result = aggregation(agg_client=agg)

    # Expect sum for x, then batch for y and m together (m is independent)
    assert agg.calls == [("batch", 1), ("batch", 2)]

    assert result["x_global"] == 10.0
    assert result["z_local"] == 11.0
    assert result["y_global"] == 11.0
    assert result["m_global"] == 3.0
    assert result["local_a"] == 13.0


def _expect_logistic(calls):
    # Expect initial totals then at least one grad/H/ll batch; allow repeats
    assert len(calls) >= 2
    assert calls[0][0] in {"batch", "sum"}  # totals
    assert calls[1][0] == "batch"


@pytest.mark.parametrize(
    "name,fn,args_builder,checker",
    [
        (
            "pearson",
            lambda agg, x, y: stats.pearson_correlation(agg, x, y, alpha=0.05),
            lambda: (
                np.array([[1.0], [2.0], [3.0]], dtype=float),
                np.array([[1.0], [0.0], [1.0]], dtype=float),
            ),
            _expect_exact([("batch", 6)]),
        ),
        (
            "ttest_one_sample",
            lambda agg, sample: stats.ttest_one_sample(
                agg, sample, mu=0.0, alpha=0.05, alternative="two-sided"
            ),
            lambda: (np.array([1.0, 2.0, 3.0], dtype=float),),
            _expect_exact([("batch", 5)]),
        ),
        (
            "ttest_paired",
            lambda agg, x, y: stats.ttest_paired(
                agg, x, y, alpha=0.05, alternative="two-sided"
            ),
            lambda: (
                np.array([1.0, 2.0, 3.0], dtype=float),
                np.array([1.0, 2.0, 4.0], dtype=float),
            ),
            _expect_exact([("batch", 3)]),
        ),
        (
            "ttest_independent",
            lambda agg, a, b: stats.ttest_independent(
                agg, a, b, alpha=0.05, alternative="two-sided"
            ),
            lambda: (
                np.array([1.0, 2.0], dtype=float),
                np.array([2.0, 3.0], dtype=float),
            ),
            _expect_exact([("batch", 2)]),
        ),
    ],
)
def test_lazy_aggregation_patterns(name, fn, args_builder, checker):
    agg = RecordingAggClient()
    args = args_builder()
    fn(agg, *args)
    checker(agg.calls)


@lazy_agg()
def aggregation_with_globals(agg_client):
    global GLOBAL_BUFFER

    first = agg_client.sum(np.array([1.0], dtype=float))
    # Mutate a global between global calls; this should force a flush.
    GLOBAL_BUFFER.append(float(np.asarray(first, dtype=float).reshape(-1)[0]))

    second = agg_client.sum(GLOBAL_INPUT)
    return float(np.asarray(first, dtype=float).reshape(-1)[0]), float(
        np.asarray(second, dtype=float).reshape(-1)[0]
    )


def test_lazy_agg_handles_global_mutation_and_globals():
    GLOBAL_BUFFER.clear()
    agg = RecordingAggClient()
    first, second = aggregation_with_globals(agg)

    assert agg.calls == [("batch", 1), ("batch", 1)]
    assert GLOBAL_BUFFER == [1.0]
    assert first == 1.0
    assert second == float(GLOBAL_INPUT.reshape(-1)[0])


@lazy_agg()
def faulty(agg_client):
    agg_client.sum(np.array([1.0], dtype=float))
    raise RuntimeError("boom")


def test_lazy_agg_preserves_original_traceback_location():
    agg = RecordingAggClient()
    with pytest.raises(RuntimeError) as excinfo:
        faulty(agg)

    source_lines, start_line = inspect.getsourcelines(faulty.__wrapped__)
    raise_line = start_line + next(
        idx for idx, line in enumerate(source_lines) if "raise RuntimeError" in line
    )

    tb = traceback.extract_tb(excinfo.value.__traceback__)
    last_frame = tb[-1]

    assert last_frame.filename == inspect.getsourcefile(faulty.__wrapped__)
    assert last_frame.lineno == raise_line


@lazy_agg()
def side_effects_flush_batches(agg_client):
    first = agg_client.sum(np.array([1.0], dtype=float))
    SIDE_EFFECT_LOG.append("touched")
    second = agg_client.sum(np.array([2.0], dtype=float))
    return (
        float(np.asarray(first, dtype=float).reshape(-1)[0]),
        float(np.asarray(second, dtype=float).reshape(-1)[0]),
    )


def test_lazy_agg_flushes_when_side_effect_present():
    SIDE_EFFECT_LOG.clear()
    agg = RecordingAggClient()
    a, b = side_effects_flush_batches(agg)

    assert agg.calls == [("batch", 1), ("batch", 1)]
    assert SIDE_EFFECT_LOG == ["touched"]
    assert a == 1.0 and b == 2.0


@lazy_agg()
def loop_batches_each_iteration(agg_client):
    outputs = []
    for val in [1.0, 2.0, 3.0]:
        a = agg_client.sum(np.array([val], dtype=float))
        b = agg_client.sum(np.array([val + 1.0], dtype=float))
        outputs.append(
            float(np.asarray(a, dtype=float).reshape(-1)[0])
            + float(np.asarray(b, dtype=float).reshape(-1)[0])
        )
    return outputs


def test_lazy_agg_batches_inside_loop_per_iteration():
    agg = RecordingAggClient()
    outputs = loop_batches_each_iteration(agg)

    assert outputs == [3.0, 5.0, 7.0]
    assert agg.calls == [("batch", 2), ("batch", 2), ("batch", 2)]


@lazy_agg()
def min_max_batching(agg_client):
    lo = agg_client.min(np.array([3.0, 1.0], dtype=float))
    hi = agg_client.max(np.array([2.0, 5.0], dtype=float))
    return (
        float(np.asarray(lo, dtype=float).reshape(-1)[0]),
        float(np.asarray(hi, dtype=float).reshape(-1)[0]),
    )


def test_lazy_agg_batches_min_and_max():
    agg = RecordingAggClient()
    lo, hi = min_max_batching(agg)

    assert agg.calls == [("batch", 2)]
    assert lo == 3.0  # returned as-is
    assert hi == 2.0


class RaisingBatchAggClient(RecordingAggClient):
    def aggregate_batch(self, ops):
        raise RuntimeError("no batch support")


@lazy_agg()
def fallback_when_batch_unavailable(agg_client):
    a = agg_client.sum(np.array([1.0], dtype=float))
    b = agg_client.min(np.array([2.0], dtype=float))
    return (
        float(np.asarray(a, dtype=float).reshape(-1)[0]),
        float(np.asarray(b, dtype=float).reshape(-1)[0]),
    )


def test_lazy_agg_falls_back_when_batch_raises():
    agg = RaisingBatchAggClient()
    a, b = fallback_when_batch_unavailable(agg)

    assert agg.calls == [("sum", 1), ("min", 1)]
    assert a == 1.0
    assert b == 2.0


@lazy_agg(agg_client_name="client")
def custom_client_name(client):
    x: float = client.sum(np.array([4.0], dtype=float))
    y: float = client.sum(np.array([5.0], dtype=float))
    return float(np.asarray(x, dtype=float).reshape(-1)[0]) + float(
        np.asarray(y, dtype=float).reshape(-1)[0]
    )


def test_lazy_agg_custom_client_name_and_annotations():
    agg = RecordingAggClient()
    total = custom_client_name(client=agg)

    assert agg.calls == [("batch", 2)]
    assert total == 9.0


@lazy_agg()
def rank_with_global_minmax(agg_client, x, y):
    # Hoist to assignments so min/max calls can be batched.
    overall_min = min([agg_client.min(x)[0], agg_client.min(y)[0]])
    overall_max = max([agg_client.max(x)[0], agg_client.max(y)[0]])
    return float(overall_min), float(overall_max)


def test_lazy_agg_handles_nested_global_minmax_calls():
    agg = RecordingAggClient()
    x = np.array([1.0, 3.0], dtype=float)
    y = np.array([5.0, 1.0], dtype=float)

    mn, mx = rank_with_global_minmax(agg, x, y)

    print(agg.calls)
    assert (mn, mx) == (1.0, 5.0)
    assert agg.calls == [("batch", 4)]
