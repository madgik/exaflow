"""Tests for NumpyAggregator collective operations.

These tests use the SimulatedAggClient + AggregationCoordinator to drive real
multi-worker barrier semantics. The coordinator aggregates per-round
contributions via ``np.stack(..., axis=0)`` (see
``simulated_agg_client._aggregate_min`` / ``_aggregate_max``), which raises a
shape error when worker contributions disagree on shape. That makes these
tests an exact reproduction of the production aggregation server's
vector-length-mismatch rejection.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _run_workers(worker_arrays, op):
    """Run ``op`` (e.g. ``aggregator.global_min``) on each worker concurrently.

    Returns the list of per-worker results in worker order. Raises any
    exception that occurred inside a worker thread.
    """
    n_workers = len(worker_arrays)
    coordinator = AggregationCoordinator(n_workers=n_workers)
    results: list = [None] * n_workers
    exceptions: list = [None] * n_workers

    def worker(idx: int, array: np.ndarray) -> None:
        try:
            client = SimulatedAggClient(worker_id=idx, coordinator=coordinator)
            aggregator = NumpyAggregator(client)
            results[idx] = getattr(aggregator, op)(array)
        except Exception as exc:  # noqa: BLE001
            exceptions[idx] = exc
            coordinator.abort(exc)

    threads = [
        threading.Thread(target=worker, args=(idx, arr))
        for idx, arr in enumerate(worker_arrays)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for exc in exceptions:
        if exc is not None:
            raise exc
    return results


def _assert_all_close(results, expected):
    """All workers must agree and match expected (with NaN-aware comparison)."""
    expected_arr = np.asarray(expected, dtype=float)
    for result in results:
        result_arr = np.asarray(result, dtype=float)
        assert result_arr.shape == expected_arr.shape
        assert np.array_equal(result_arr, expected_arr, equal_nan=True)


class TestGlobalMinWithEmptyShard:
    """``global_min`` must produce a length-K sentinel when input is shape (0, K)."""

    def test_one_empty_worker_among_2d_contributors(self):
        # Two non-empty workers + one with shape (0, 3). Without the fix the
        # empty worker submits a 0-D scalar, np.stack inside the coordinator
        # raises a shape error, and the request fails.
        worker_arrays = [
            np.array([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0]]),  # axis-0 min: [1, 5, 9]
            np.array([[4.0, 3.0, 11.0]]),  # axis-0 min: [4, 3, 11]
            np.empty((0, 3)),  # sentinel:   [inf, inf, inf]
        ]

        results = _run_workers(worker_arrays, op="global_min")

        _assert_all_close(results, expected=[1.0, 3.0, 9.0])

    def test_all_workers_empty_2d_returns_all_nan(self):
        # When every worker is empty, every contribution is +inf — global min
        # is +inf, which the aggregator converts back to NaN.
        worker_arrays = [np.empty((0, 3)) for _ in range(3)]

        results = _run_workers(worker_arrays, op="global_min")

        _assert_all_close(results, expected=[np.nan, np.nan, np.nan])

    def test_one_empty_worker_among_1d_contributors(self):
        # 1-D inputs reduce to a 0-D scalar; the sentinel for empty 1-D is
        # also 0-D, so shape stays consistent across workers.
        worker_arrays = [
            np.array([3.0, 1.0, 7.0]),  # nanmin: 1.0
            np.array([5.0, 2.0, 8.0]),  # nanmin: 2.0
            np.empty((0,)),  # sentinel: inf (0-D)
        ]

        results = _run_workers(worker_arrays, op="global_min")

        for result in results:
            assert np.asarray(result).shape == ()
            assert float(result) == pytest.approx(1.0)

    def test_one_empty_worker_among_3d_contributors(self):
        # Verifies the fix handles ranks > 2 — sentinel shape must match the
        # axis-0-reduced shape, here (2, 4).
        worker_arrays = [
            np.full((3, 2, 4), 5.0),  # nanmin: all 5s
            np.full((1, 2, 4), 2.0),  # nanmin: all 2s
            np.empty((0, 2, 4)),  # sentinel: all-inf (2, 4)
        ]

        results = _run_workers(worker_arrays, op="global_min")

        _assert_all_close(results, expected=np.full((2, 4), 2.0))


class TestGlobalMaxWithEmptyShard:
    """``global_max`` must produce a length-K sentinel when input is shape (0, K)."""

    def test_one_empty_worker_among_2d_contributors(self):
        worker_arrays = [
            np.array([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0]]),  # axis-0 max: [2, 6, 10]
            np.array([[4.0, 3.0, 11.0]]),  # axis-0 max: [4, 3, 11]
            np.empty((0, 3)),  # sentinel:   [-inf, -inf, -inf]
        ]

        results = _run_workers(worker_arrays, op="global_max")

        _assert_all_close(results, expected=[4.0, 6.0, 11.0])

    def test_all_workers_empty_2d_returns_all_nan(self):
        worker_arrays = [np.empty((0, 3)) for _ in range(3)]

        results = _run_workers(worker_arrays, op="global_max")

        _assert_all_close(results, expected=[np.nan, np.nan, np.nan])

    def test_one_empty_worker_among_1d_contributors(self):
        worker_arrays = [
            np.array([3.0, 1.0, 7.0]),  # nanmax: 7.0
            np.array([5.0, 2.0, 8.0]),  # nanmax: 8.0
            np.empty((0,)),  # sentinel: -inf (0-D)
        ]

        results = _run_workers(worker_arrays, op="global_max")

        for result in results:
            assert np.asarray(result).shape == ()
            assert float(result) == pytest.approx(8.0)
