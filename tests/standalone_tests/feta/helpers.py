from threading import Condition
from threading import Event

import numpy as np
import pyarrow as pa

from exaflow.aggregation_serialization import bytes_to_ndarray
from exaflow.controller.services.feta.runtime import FetaRuntime
from exaflow.feta.lifecycle import ContextClosedError
from exaflow.worker.feta.service import FetaWorkerService


class MockAggregation:
    """Independent mock aggregation server for threaded federation simulations."""

    def __init__(self):
        self.lock = Condition()
        self.rounds = {}
        self.events = []
        self.fetch_started = Event()

    def prepare(self, request):
        with self.lock:
            key = (request.context_id, request.step)
            assert key not in self.rounds
            self.rounds[key] = (set(request.worker_ids), {})
            self.events.append(("prepare", request.step))

    def submit(self, request):
        with self.lock:
            expected, values = self.rounds[(request.context_id, request.step)]
            assert request.worker_id in expected and request.worker_id not in values
            values[request.worker_id] = bytes_to_ndarray(request.value.arrow)
            self.events.append(("submit", request.step))
            self.lock.notify_all()

    def fetch(self, context_id, step):
        with self.lock:
            key = (context_id, step)
            self.fetch_started.set()
            ready = self.lock.wait_for(
                lambda: (
                    key not in self.rounds
                    or set(self.rounds[key][1]) == self.rounds[key][0]
                ),
                timeout=2,
            )
            if key not in self.rounds:
                raise ContextClosedError("Mock aggregation cancelled.")
            assert ready, "Mock aggregation timed out"
            expected, values = self.rounds.pop(key)
            assert set(values) == expected
            self.events.append(("fetch", step))
            return np.sum([values[key] for key in sorted(values)], axis=0)

    def cleanup(self, context_id):
        with self.lock:
            self.rounds = {
                key: value for key, value in self.rounds.items() if key[0] != context_id
            }
            self.lock.notify_all()

    def close(self):
        pass

    def cancel(self):
        pass


class InlineWorker:
    def __init__(self, worker_id, service):
        self.worker_id = worker_id
        self.service = service
        self.operations = []

    def initialize(self, request):
        return self.service.initialize(request)

    def execute(self, request):
        self.operations.append(request.operation.type)
        return self.service.execute(request)

    def cleanup(self, context_id):
        self.service.store.cleanup(context_id)

    def close(self):
        pass

    def cancel(self):
        pass


def federation(parts, minimum_rows=0, settings=None):
    aggregation = MockAggregation()
    workers = []
    for idx, frame in enumerate(parts):
        loader = lambda data, include_dataset=False, frame=frame: pa.Table.from_pandas(
            frame[data.variables], preserve_index=False
        )
        service = FetaWorkerService(
            f"worker-{idx}",
            settings=settings,
            minimum_rows=minimum_rows,
            loader=loader,
            aggregation_factory=lambda address: aggregation,
        )
        workers.append(InlineWorker(service.worker_id, service))
    return FetaRuntime(workers, aggregation, settings=settings), workers, aggregation
