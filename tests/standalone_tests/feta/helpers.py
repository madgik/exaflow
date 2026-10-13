from threading import Condition
from threading import Event

import numpy as np
import pyarrow as pa

from exaflow.controller.services.feta.runtime import FetaRuntime
from exaflow.feta.lifecycle import ContextClosedError
from exaflow.feta.operations import decode_tensor
from exaflow.protos.feta import feta_pb2 as pb
from exaflow.worker.feta.service import FetaWorkerService


class MockAggregation:
    """Independent mock aggregation server for threaded federation simulations."""

    def __init__(self):
        self.lock = Condition()
        self.rounds = {}
        self.events = []
        self.kinds = {}
        self.dtypes = {}
        self.fetch_started = Event()

    def prepare(self, request):
        with self.lock:
            key = (request.context_id, request.step)
            assert key not in self.rounds
            self.rounds[key] = (set(request.worker_ids), {})
            self.kinds[key] = request.kind
            self.dtypes[key] = request.dtype
            self.events.append(("prepare", request.step))

    def submit(self, request):
        with self.lock:
            expected, values = self.rounds[(request.context_id, request.step)]
            assert request.worker_id in expected and request.worker_id not in values
            values[request.worker_id] = decode_tensor(request.value, 100 * 1024 * 1024)
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
            arrays = [values[worker] for worker in sorted(values)]
            kind = self.kinds.pop(key)
            dtype = self.dtypes.pop(key)
            if kind == pb.UNION_VALUES:
                return np.unique(np.concatenate(arrays).astype(dtype))
            if kind == pb.MIN_VALUES:
                return np.min(arrays, axis=0)
            if kind == pb.MAX_VALUES:
                return np.max(arrays, axis=0)
            return np.sum(arrays, axis=0)

    def cleanup(self, context_id):
        with self.lock:
            self.rounds = {
                key: value for key, value in self.rounds.items() if key[0] != context_id
            }
            self.kinds = {k: v for k, v in self.kinds.items() if k[0] != context_id}
            self.dtypes = {k: v for k, v in self.dtypes.items() if k[0] != context_id}
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
