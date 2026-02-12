from __future__ import annotations

import threading
from abc import ABC
from abc import abstractmethod

import numpy as np

from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


class FederatedAlgorithmTest(ABC):
    """Base test case to compare federated vs centralized computation."""

    @abstractmethod
    def compute_centralized_result(self, X, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compare(self, federated_output, centralized_output, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def test_federated_algorithm_with_one_worker(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def test_federated_algorithm_with_multiple_workers(self, **kwargs):
        raise NotImplementedError

    def _outputs_equal(self, left, right):
        if left is right:
            return True
        if left is None or right is None:
            return left is right

        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return np.allclose(
                np.asarray(left), np.asarray(right), atol=1e-8, rtol=1e-8
            )

        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                return False
            return all(
                self._outputs_equal(left[key], right[key]) for key in left.keys()
            )

        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                return False
            return all(self._outputs_equal(l, r) for l, r in zip(left, right))

        if hasattr(left, "__dict__") and hasattr(right, "__dict__"):
            left_vars = {k: v for k, v in vars(left).items() if k != "agg_client"}
            right_vars = {k: v for k, v in vars(right).items() if k != "agg_client"}
            return self._outputs_equal(left_vars, right_vars)

        return left == right

    def _validate_federated_outputs(self, federated_outputs):
        baseline = federated_outputs[0]
        for federated_output in federated_outputs[1:]:
            assert self._outputs_equal(baseline, federated_output)

    def _split_inputs(self, X, y, n_workers: int):
        return _split_federated_inputs(X, y, n_workers)

    def run_comparison(
        self,
        *,
        X,
        y,
        n_workers: int = 5,
        **kwargs,
    ):
        x_parts, y_parts, X_full, y_full = self._split_inputs(X, y, n_workers)

        def _worker_fn(worker_id, agg_client):
            return self.compute_federated_result(
                x_parts[worker_id],
                y_parts[worker_id],
                agg_client=agg_client,
                **kwargs,
            )

        federated_outputs = _simulate_federated_execution(n_workers, _worker_fn)
        centralized_output = self.compute_centralized_result(X_full, y_full, **kwargs)
        compare_kwargs = dict(kwargs)
        compare_kwargs["n_workers"] = n_workers
        self._validate_federated_outputs(federated_outputs)
        self.compare(federated_outputs[0], centralized_output, **compare_kwargs)


def _split_federated_inputs(X, y, n_workers: int):
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and isinstance(X, pd.DataFrame):
        indices = np.array_split(np.arange(len(X)), n_workers)
        x_parts = [X.iloc[idx] for idx in indices]
        X_full = X
    else:
        X_full = np.asarray(X)
        x_parts = np.array_split(X_full, n_workers)

    if pd is not None and isinstance(y, pd.Series):
        indices = np.array_split(np.arange(len(y)), n_workers)
        y_parts = [y.iloc[idx] for idx in indices]
        y_full = y
    else:
        y_full = np.asarray(y)
        y_parts = np.array_split(y_full, n_workers)

    return x_parts, y_parts, X_full, y_full


def _simulate_federated_execution(n_workers: int, worker_fn):
    """Run N workers in threads with a shared aggregation coordinator.

    worker_fn signature: (worker_id: int, agg_client: SimulatedAggClient) -> result
    """
    coordinator = AggregationCoordinator(n_workers=n_workers)
    results = [None] * n_workers
    errors = [None] * n_workers

    def _run(worker_id: int):
        try:
            agg_client = SimulatedAggClient(
                worker_id=worker_id, coordinator=coordinator
            )
            results[worker_id] = worker_fn(worker_id, agg_client)
        except Exception as exc:  # pragma: no cover - best-effort error capture
            coordinator.abort(exc)
            errors[worker_id] = exc

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(n_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for error in errors:
        if error is not None:
            raise error

    return results
