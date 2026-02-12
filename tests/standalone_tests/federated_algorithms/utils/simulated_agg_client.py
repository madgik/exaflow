import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
from typing import Sequence

import numpy as np

from exaflow.algorithms.federated.utils.agg_client import AggregationClient


def _aggregate_sum(values: Iterable) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float) for value in values]
    if not arrays:
        return np.asarray([], dtype=float)
    return np.sum(np.stack(arrays, axis=0), axis=0)


def _aggregate_min(values: Iterable) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float) for value in values]
    if not arrays:
        return np.asarray([], dtype=float)
    return np.min(np.stack(arrays, axis=0), axis=0)


def _aggregate_max(values: Iterable) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float) for value in values]
    if not arrays:
        return np.asarray([], dtype=float)
    return np.max(np.stack(arrays, axis=0), axis=0)


def _aggregate_union(values: Iterable[Sequence[object]]) -> Sequence[object]:
    combined = set()
    for value in values:
        combined.update(val for val in value if val is not None)
    return sorted(combined)


@dataclass(frozen=True)
class _RoundKey:
    op: str
    index: int


class AggregationCoordinator:
    """Thread-safe coordinator that blocks until all workers submit per round."""

    def __init__(self, n_workers: int):
        if n_workers <= 0:
            raise ValueError("n_workers must be positive")
        self.n_workers = n_workers
        self._cond = threading.Condition()
        self._buckets = defaultdict(dict)
        self._results = {}
        self._aborted = False
        self._abort_exc = None

    def abort(self, exc: Exception) -> None:
        with self._cond:
            if self._aborted:
                return
            self._aborted = True
            self._abort_exc = exc
            self._cond.notify_all()

    def submit(self, *, op: str, round_idx: int, worker_id: int, value):
        key = _RoundKey(op=op, index=round_idx)
        with self._cond:
            if self._aborted:
                raise self._abort_exc
            if key in self._results:
                return self._results[key]

            bucket = self._buckets[key]
            bucket[worker_id] = value
            if len(bucket) < self.n_workers:
                while key not in self._results and not self._aborted:
                    self._cond.wait()
                if self._aborted and key not in self._results:
                    raise self._abort_exc
                return self._results[key]

            if op == "sum":
                result = _aggregate_sum(bucket.values())
            elif op == "min":
                result = _aggregate_min(bucket.values())
            elif op == "max":
                result = _aggregate_max(bucket.values())
            elif op == "union":
                result = _aggregate_union(bucket.values())
            else:
                raise ValueError(f"Unsupported op: {op}")

            self._results[key] = result
            self._cond.notify_all()
            return result


class SimulatedAggClient(AggregationClient):
    """Aggregation client that synchronizes with other workers in threads."""

    def __init__(self, *, worker_id: int, coordinator: AggregationCoordinator):
        self._worker_id = worker_id
        self._coordinator = coordinator
        self._rounds = defaultdict(int)

    def _next_round(self, op: str) -> int:
        round_idx = self._rounds[op]
        self._rounds[op] += 1
        return round_idx

    def sum(self, values):
        return self._coordinator.submit(
            op="sum",
            round_idx=self._next_round("sum"),
            worker_id=self._worker_id,
            value=values,
        )

    def min(self, values):
        return self._coordinator.submit(
            op="min",
            round_idx=self._next_round("min"),
            worker_id=self._worker_id,
            value=values,
        )

    def max(self, values):
        return self._coordinator.submit(
            op="max",
            round_idx=self._next_round("max"),
            worker_id=self._worker_id,
            value=values,
        )

    def union(self, values):
        return self._coordinator.submit(
            op="union",
            round_idx=self._next_round("union"),
            worker_id=self._worker_id,
            value=values,
        )
