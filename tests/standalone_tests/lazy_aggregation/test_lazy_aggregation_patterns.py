import inspect
import traceback

import numpy as np
import pytest

from exaflow.worker.exareme3.lazy_aggregation import lazy_agg


class RecordingAggClient:
    def __init__(self):
        self.calls = []

    def _aggregate_batch(self, ops):
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

    def union(self, value):
        self.calls.append(("union", np.asarray(value).size))
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

    # m is independent: it goes out with x, then y (which depends on x) alone
    assert agg.calls == [("batch", 2), ("batch", 1)]

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
def side_effect_between_aggregations(agg_client):
    first = agg_client.sum(np.array([1.0], dtype=float))
    SIDE_EFFECT_LOG.append("touched")
    second = agg_client.sum(np.array([2.0], dtype=float))
    return (
        float(np.asarray(first, dtype=float).reshape(-1)[0]),
        float(np.asarray(second, dtype=float).reshape(-1)[0]),
    )


def test_lazy_agg_batches_across_side_effects_that_ignore_results():
    SIDE_EFFECT_LOG.clear()
    agg = RecordingAggClient()
    a, b = side_effect_between_aggregations(agg)

    assert agg.calls == [("batch", 2)]
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


class NoBatchAggClient(RecordingAggClient):
    _aggregate_batch = None


class RaisingBatchAggClient(RecordingAggClient):
    def _aggregate_batch(self, ops):
        raise RuntimeError("no batch support")


@lazy_agg()
def fallback_when_batch_unavailable(agg_client):
    a = agg_client.sum(np.array([1.0], dtype=float))
    b = agg_client.min(np.array([2.0], dtype=float))
    return (
        float(np.asarray(a, dtype=float).reshape(-1)[0]),
        float(np.asarray(b, dtype=float).reshape(-1)[0]),
    )


def test_lazy_agg_uses_individual_calls_without_batch_support():
    agg = NoBatchAggClient()
    a, b = fallback_when_batch_unavailable(agg)

    assert agg.calls == [("sum", 1), ("min", 1)]
    assert a == 1.0
    assert b == 2.0


def test_lazy_agg_propagates_batch_errors():
    with pytest.raises(RuntimeError, match="no batch support"):
        fallback_when_batch_unavailable(RaisingBatchAggClient())


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


def _global_mean(client, values):
    total = client.sum(np.array([values.sum()], dtype=float))
    count = client.sum(np.array([float(values.size)], dtype=float))
    return np.asarray(total)[0] / np.asarray(count)[0]


@lazy_agg()
def means_through_helpers(agg_client, x, y):
    mean_x = float(_global_mean(agg_client, x))
    mean_y = float(_global_mean(agg_client, y))
    return mean_x, mean_y


def test_lazy_agg_batches_across_helper_calls():
    agg = RecordingAggClient()
    result = means_through_helpers(agg, np.array([1.0, 3.0]), np.array([2.0, 4.0, 6.0]))

    assert result == (2.0, 4.0)
    assert agg.calls == [("batch", 4)]


def _levels(client, values):
    if values.dtype.kind == "f":
        result = client.union(np.unique(values[~np.isnan(values)]))
        return np.array(result)
    else:
        result = client.union(np.unique(values))
        return np.array(result)


@lazy_agg()
def levels_of_two_columns(agg_client, a, b):
    return _levels(agg_client, a).tolist(), _levels(agg_client, b).tolist()


def test_lazy_agg_batches_helpers_returning_from_branches():
    agg = RecordingAggClient()
    result = levels_of_two_columns(
        agg, np.array([2.0, np.nan, 1.0]), np.array(["b", "a", "b"])
    )

    assert result == ([1.0, 2.0], ["a", "b"])
    assert agg.calls == [("batch", 2)]


@lazy_agg()
def aggregations_around_branch(agg_client, x, use_squares):
    if use_squares:
        squares = np.asarray(agg_client.sum(x * x), dtype=float)
    else:
        squares = np.zeros(1)
    n_obs = float(np.asarray(agg_client.sum(np.array([float(x.size)])))[0])
    return squares.tolist(), n_obs


def test_lazy_agg_batches_across_pure_branches():
    agg = RecordingAggClient()
    result = aggregations_around_branch(agg, np.array([1.0, 2.0]), True)

    assert result == ([1.0, 4.0], 2.0)
    assert agg.calls == [("batch", 2)]


@lazy_agg()
def aggregations_in_try(agg_client, x):
    try:
        a = float(np.asarray(agg_client.sum(x))[0])
        b = float(np.asarray(agg_client.sum(x * 2.0))[0])
    except ValueError:
        a = b = 0.0
    return a, b


def test_lazy_agg_does_not_reorder_inside_try():
    agg = RecordingAggClient()

    assert aggregations_in_try(agg, np.array([1.0])) == (1.0, 2.0)
    assert agg.calls == [("batch", 1), ("batch", 1)]


@lazy_agg()
def conditional_aggregation(agg_client, x, flag):
    a = agg_client.sum(x)
    b = agg_client.sum(x * 2.0) if flag else None
    return a, b


@pytest.mark.parametrize("flag,expected_calls", [(False, 1), (True, 2)])
def test_lazy_agg_never_runs_conditional_aggregations_unconditionally(
    flag, expected_calls
):
    agg = RecordingAggClient()
    conditional_aggregation(agg, np.array([1.0]), flag)

    assert agg.calls == [("batch", expected_calls)]


def _double_in_place(values):
    values *= 2.0
    return values


@lazy_agg()
def mutation_while_queued(agg_client, x):
    total = float(np.asarray(agg_client.sum(x))[0])
    doubled = float(np.asarray(_double_in_place(x)).sum())
    return total, doubled


def test_lazy_agg_sends_values_as_they_were_when_submitted():
    agg = RecordingAggClient()
    x = np.array([1.0, 2.0])

    # _double_in_place runs before the batch is sent; the queued copy is sent.
    assert mutation_while_queued(agg, x) == (1.0, 6.0)
    assert x.tolist() == [2.0, 4.0]


@lazy_agg()
def raises_before_result_is_needed(agg_client, x, divisor):
    ratio = 1.0 / divisor
    total = agg_client.sum(x)
    return ratio, total


def test_lazy_agg_discards_queued_aggregations_on_exception():
    agg = RecordingAggClient()

    with pytest.raises(ZeroDivisionError):
        raises_before_result_is_needed(agg, np.array([1.0]), 0.0)
    assert agg.calls == []


@lazy_agg()
def local_reductions(agg_client, x):
    column_sums = x.sum(0)
    total = agg_client.sum(column_sums)
    return column_sums.tolist(), np.asarray(total).tolist()


def test_lazy_agg_leaves_non_aggregation_reductions_alone():
    agg = RecordingAggClient()

    assert local_reductions(agg, np.array([[1.0, 2.0], [3.0, 4.0]])) == (
        [4.0, 6.0],
        [4.0, 6.0],
    )
    assert agg.calls == [("batch", 1)]
