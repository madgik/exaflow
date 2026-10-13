import math
import time
from dataclasses import dataclass
from dataclasses import field
from threading import Condition
from threading import RLock

import numpy as np

from exaflow.aggregation_serialization import ndarray_to_bytes
from exaflow.feta.lifecycle import ContextClosedError
from exaflow.feta.lifecycle import ResourceLimitError
from exaflow.feta.operations import decode_array
from exaflow.feta.rpc import abort
from exaflow.feta.settings import FetaSettings
from exaflow.protos.feta import feta_pb2 as pb
from exaflow.protos.feta import feta_pb2_grpc as rpc


@dataclass
class AggregationSession:
    deadline: float
    workers: frozenset
    condition: Condition = field(default_factory=Condition)
    last_step: int = 0
    step: int = 0
    shape: tuple = ()
    dtype: str = ""
    contributions: dict = field(default_factory=dict)
    submitted: set = field(default_factory=set)
    result: object = None
    error: Exception | None = None
    closed: bool = False
    fetching: bool = False


class FetaAggregationService(rpc.FetaAggregationServicer):
    """Prepare a step, acknowledge worker submissions, then serve one fetch."""

    def __init__(self, settings=None, wait_timeout=60, clock=time.monotonic):
        self.settings = settings or FetaSettings()
        self.wait_timeout = wait_timeout
        self.clock = clock
        self.sessions = {}
        self.lock = RLock()

    def _get(self, context_id):
        with self.lock:
            session = self.sessions.get(context_id)
        if session is None or session.closed:
            raise ContextClosedError("FETA aggregation context is closed.")
        if self.clock() >= session.deadline:
            self.cleanup(context_id)
            raise ContextClosedError("FETA aggregation context expired.")
        return session

    def prepare(self, request):
        workers = frozenset(request.worker_ids)
        shape = tuple(request.shape)
        if not request.context_id or len(request.context_id) > 128:
            raise ValueError("Invalid FETA context ID.")
        if not workers or len(workers) != len(request.worker_ids):
            raise ValueError("Aggregation needs distinct participating worker IDs.")
        if any(not worker or len(worker) > 128 for worker in workers):
            raise ValueError("Invalid participating worker ID.")
        try:
            dtype = np.dtype(request.dtype)
        except TypeError as exc:
            raise ValueError("Invalid aggregation dtype.") from exc
        if (
            dtype.kind not in "iuf"
            or dtype.itemsize > 8
            or not dtype.isnative
            or (dtype.kind == "f" and dtype.itemsize < 4)
        ):
            raise ValueError("Invalid numerical aggregation dtype.")
        if len(shape) > 2 or any(n < 0 for n in shape):
            raise ValueError("Invalid aggregation shape.")
        size = math.prod(shape) * dtype.itemsize
        if size > self.settings.max_tensor_bytes or (
            size * (len(workers) + 1) > self.settings.max_context_bytes
        ):
            raise ResourceLimitError("FETA aggregation memory limit exceeded.")
        with self.lock:
            session = self.sessions.get(request.context_id)
            if session is None:
                if request.step != 1:
                    raise ValueError("First FETA aggregation step must be one.")
                if len(self.sessions) >= self.settings.max_contexts:
                    raise ResourceLimitError("FETA aggregation context limit exceeded.")
                session = AggregationSession(
                    self.clock() + self.settings.context_lifetime_minutes * 60,
                    workers,
                )
                self.sessions[request.context_id] = session
        session = self._get(request.context_id)
        with session.condition:
            if session.closed:
                raise ContextClosedError("FETA aggregation context is closed.")
            if session.error is not None:
                raise session.error
            if session.step or request.step != session.last_step + 1:
                raise ValueError("FETA aggregation steps must be sequential.")
            if workers != session.workers:
                raise ValueError("FETA participation cannot change during execution.")
            session.step = request.step
            session.shape = shape
            session.dtype = str(dtype)

    def submit(self, request):
        session = self._get(request.context_id)
        try:
            value = decode_array(request.value.arrow, self.settings.max_tensor_bytes)
            with session.condition:
                if session.closed:
                    raise ContextClosedError("FETA aggregation context is closed.")
                if session.error is not None:
                    raise session.error
                if not session.step or request.step != session.step:
                    raise ValueError("Contribution targets an unprepared aggregation.")
                if request.worker_id not in session.workers:
                    raise ValueError("Contribution from an unexpected worker.")
                if request.worker_id in session.submitted:
                    raise ValueError("Duplicate FETA contribution.")
                if value.shape != session.shape or str(value.dtype) != session.dtype:
                    raise ValueError(
                        "Contribution shape/dtype differs from preparation."
                    )
                session.contributions[request.worker_id] = value
                session.submitted.add(request.worker_id)
                if len(session.contributions) != len(session.workers):
                    return
                # Stable worker order; do not hold a lock during NumPy computation.
                values = [session.contributions[key] for key in sorted(session.workers)]
            result = np.zeros(session.shape, dtype=session.dtype)
            for value in values:
                np.add(result, value, out=result)
            with session.condition:
                if session.closed or self.clock() >= session.deadline:
                    raise ContextClosedError(
                        "FETA aggregation closed during computation."
                    )
                if session.error is not None:
                    raise session.error
                session.result = result
                session.contributions.clear()
                session.condition.notify_all()
        except Exception as exc:
            with session.condition:
                session.error = exc
                session.condition.notify_all()
            raise

    def fetch(self, request, active=lambda: True):
        session = self._get(request.context_id)
        end = min(session.deadline, self.clock() + self.wait_timeout)
        with session.condition:
            if not session.step or request.step != session.step:
                raise ValueError("Fetch targets an unprepared aggregation.")
            if session.fetching:
                raise ValueError("Only one FETA result fetch may wait per step.")
            session.fetching = True
            try:
                while (
                    session.result is None
                    and session.error is None
                    and not session.closed
                ):
                    if not active():
                        session.error = ContextClosedError(
                            "FETA result fetch cancelled."
                        )
                        break
                    remaining = end - self.clock()
                    if remaining <= 0:
                        session.error = TimeoutError(
                            "Timed out waiting for FETA contributions."
                        )
                        break
                    session.condition.wait(timeout=min(remaining, 0.1))
                if session.closed or self.clock() >= session.deadline:
                    raise ContextClosedError(
                        "FETA aggregation context closed or expired."
                    )
                if session.error is not None:
                    raise session.error
                value = session.result
                session.result = None
                session.submitted.clear()
                session.last_step = session.step
                session.step = 0
                return value
            finally:
                session.fetching = False

    def cleanup(self, context_id):
        with self.lock:
            session = self.sessions.pop(context_id, None)
        if session is not None:
            with session.condition:
                session.closed = True
                session.result = None
                session.contributions.clear()
                session.submitted.clear()
                session.condition.notify_all()

    def expire(self):
        with self.lock:
            expired = [
                key for key, s in self.sessions.items() if self.clock() >= s.deadline
            ]
        for key in expired:
            self.cleanup(key)
        return len(expired)

    def Prepare(self, request, context):
        try:
            self.prepare(request)
            return pb.Ack()
        except Exception as exc:
            abort(context, exc)

    def Submit(self, request, context):
        try:
            self.submit(request)
            return pb.Ack()
        except Exception as exc:
            abort(context, exc)

    def Fetch(self, request, context):
        try:
            value = self.fetch(request, context.is_active)
            return pb.TensorValue(arrow=ndarray_to_bytes(value))
        except Exception as exc:
            abort(context, exc)

    def Cleanup(self, request, context):
        self.cleanup(request.context_id)
        return pb.Ack()
