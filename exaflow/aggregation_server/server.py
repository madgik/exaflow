"""gRPC aggregation server: combines vectors from the workers of one request.

Lifecycle of a request
----------------------
- The controller calls `Configure(request_id, num_of_workers)` before running
  the algorithm and `Cleanup(request_id)` when the request ends, in a `finally`
  block.
- Each worker calls `Aggregate(request_id, step, operations)` from its UDF.
  `step` is numbered by the worker's client (1, 2, 3, ...). `operations` holds
  one or more `(aggregation_type, tensor)` pairs. Numeric types (`SUM`, `MIN`,
  `MAX`) send Arrow tensors; `UNION` sends JSON-encoded values.
- A worker that drops out (too little data, bad input) calls `Unregister`, which
  lowers the number of inputs each later step waits for.

Synchronization
---------------
Each request has an `AggregationContext` guarded by one `Condition`. It keeps
one slot per step number that is still in progress:
- A worker's `Aggregate` call adds its inputs to the slot for its step,
  creating the slot if it is the first to arrive.
- When the slot has inputs from all `expected_workers`, the result is computed
  and every waiting worker wakes up and reads it.
- After the last read the slot is deleted. A worker that finishes a step early
  can already start the next one; it gets its own slot.
Every wait is bounded by `config.max_wait_for_aggregation_inputs`.

Errors
------
The first error (mismatched operations, bad payload, timeout) ends the request:
every waiting worker and every later call for the request gets that same error.
The controller then ends the request and calls `Cleanup`.

`Cleanup` removes the context and ends the request with ABORTED, so workers
still waiting on a step, and calls that fetched the context just before it was
removed, return at once instead of at the timeout. Late Aggregate/Unregister
calls after Cleanup also get ABORTED (not INVALID_ARGUMENT, which is reserved
for never-configured ids). If `Cleanup` never arrives (for example the
controller crashed), `Configure` evicts contexts with no `Aggregate` or
`Unregister` call for `STALE_CONTEXT_TTL_SECONDS`.

`AggregationContext` raises plain Python exceptions and only the servicer
methods call `context.abort()`, mapping them with `_grpc_status()`. gRPC's
`abort()` raises a bare `Exception`, so it must never run inside a
`try/except Exception` block or its status and details are lost.
"""

import copy
import logging
import threading
import time
from concurrent import futures
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import grpc
import numpy as np
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from exaflow.aggregation_serialization import bytes_to_ndarray
from exaflow.aggregation_serialization import bytes_to_values
from exaflow.aggregation_serialization import ndarray_to_bytes
from exaflow.aggregation_serialization import values_to_bytes
from exaflow.aggregation_server import config
from exaflow.aggregation_types import AggregationType
from exaflow.grpc_options import grpc_message_options
from exaflow.protos.aggregation_server.aggregation_server_pb2 import AggregateResponse
from exaflow.protos.aggregation_server.aggregation_server_pb2 import CleanupResponse
from exaflow.protos.aggregation_server.aggregation_server_pb2 import ConfigureResponse
from exaflow.protos.aggregation_server.aggregation_server_pb2 import Status
from exaflow.protos.aggregation_server.aggregation_server_pb2 import UnregisterResponse
from exaflow.protos.aggregation_server.aggregation_server_pb2_grpc import (
    AggregationServerServicer,
)
from exaflow.protos.aggregation_server.aggregation_server_pb2_grpc import (
    add_AggregationServerServicer_to_server,
)

logger = logging.getLogger("AggregationServer")

NUMERIC_KINDS = {"i", "u", "f"}
# Numeric aggregation types. UNION is the only other one and works on
# JSON-encoded values instead of Arrow tensors.
REDUCERS = {
    AggregationType.SUM: np.add,
    AggregationType.MIN: np.minimum,
    AggregationType.MAX: np.maximum,
}
# Contexts with no Aggregate or Unregister call for this long are treated as
# abandoned (their Cleanup never arrived) and evicted on the next Configure.
# A running request never idles near this: each aggregation wait is capped at
# max_wait_for_aggregation_inputs and each UDF call at the worker tasks timeout.
STALE_CONTEXT_TTL_SECONDS = 24 * 60 * 60


def _result_dtype(
    aggregation_type: AggregationType, dtypes: list[np.dtype]
) -> np.dtype:
    dtype = np.result_type(*dtypes)
    if all(input_dtype.kind in "iu" for input_dtype in dtypes):
        if dtype.kind not in "iu":
            raise ValueError(
                "Integer aggregation inputs cannot be safely promoted to a "
                "fixed-width integer dtype."
            )
        if aggregation_type == AggregationType.SUM:
            # Sum small integers in 64 bits so the total cannot wrap around.
            dtype = np.dtype(f"{dtype.kind}8")
    return dtype


def _decode_operations(request) -> Tuple[List[AggregationType], List[np.ndarray]]:
    if not request.operations:
        raise ValueError(f"request_id='{request.request_id}' sent no operations")
    aggregation_types = []
    payloads = []
    for op in request.operations:
        try:
            aggregation_type = AggregationType(op.aggregation_type)
        except ValueError:
            raise ValueError(
                f"Unsupported aggregation type '{op.aggregation_type}'"
            ) from None
        if not op.tensor:
            raise ValueError(
                f"request_id='{request.request_id}' missing tensor payload"
            )
        if aggregation_type == AggregationType.UNION:
            payload = bytes_to_values(op.tensor)
        else:
            payload = bytes_to_ndarray(op.tensor)
            if payload.dtype.kind not in NUMERIC_KINDS:
                raise ValueError(
                    "Numeric aggregation requires numeric tensor payloads."
                )
        aggregation_types.append(aggregation_type)
        payloads.append(payload)
    return aggregation_types, payloads


class AggregationCancelledError(Exception):
    """The request was cleaned up while a worker was still waiting on it."""


def _grpc_status(exc: Exception) -> grpc.StatusCode:
    if isinstance(exc, AggregationCancelledError):
        return grpc.StatusCode.ABORTED
    if isinstance(exc, TimeoutError):
        return grpc.StatusCode.DEADLINE_EXCEEDED
    if isinstance(exc, ValueError):
        return grpc.StatusCode.INVALID_ARGUMENT
    return grpc.StatusCode.INTERNAL


@dataclass
class _Step:
    aggregation_types: List[AggregationType]
    # inputs[i] holds one vector per worker for operation i.
    inputs: List[List[np.ndarray]]
    result: Optional[List[np.ndarray]] = None
    reads: int = 0


class AggregationContext:
    """Aggregation state of one request, guarded by `condition`."""

    def __init__(self, request_id: str, expected_workers: int):
        self.request_id = request_id
        self.expected_workers = expected_workers
        self.steps: Dict[int, _Step] = {}
        self.last_completed_step = 0
        self.error: Optional[Exception] = None
        self.last_activity = time.monotonic()
        self.condition = threading.Condition()

    def fail(self, exc: Exception, *, log: bool = True) -> None:
        """End the request: every waiting and later call raises the first error."""
        with self.condition:
            if self.error is None:
                self.error = exc
                if log:
                    logger.error(
                        f"[AGGREGATE] request_id='{self.request_id}' failed: {exc}",
                        exc_info=exc,
                    )
            self.condition.notify_all()

    def _raise_error(self) -> None:
        # Raise a copy: every caller shares the stored error, and each raise of
        # the same object would extend its traceback.
        raise copy.copy(self.error) from self.error

    def aggregate(
        self,
        step_no: int,
        aggregation_types: List[AggregationType],
        payloads: List[np.ndarray],
        reduce_fn,
    ) -> List[np.ndarray]:
        """Add one worker's inputs for `step_no` and wait for the combined result."""
        with self.condition:
            try:
                if self.error is not None:
                    self._raise_error()
                if step_no <= self.last_completed_step:
                    raise ValueError(
                        f"Step {step_no} already completed for request_id='{self.request_id}'"
                    )
                step = self.steps.setdefault(
                    step_no, _Step(aggregation_types, [[] for _ in aggregation_types])
                )
                self._add_inputs(step, step_no, aggregation_types, payloads)
                logger.info(
                    "[AGGREGATE] request_id='%s' step %d: %d/%d workers reported",
                    self.request_id,
                    step_no,
                    len(step.inputs[0]),
                    self.expected_workers,
                )
                self.compute_complete_steps(reduce_fn)

                done = self.condition.wait_for(
                    lambda: step.result is not None or self.error is not None,
                    timeout=config.max_wait_for_aggregation_inputs,
                )
                if self.error is not None:
                    self._raise_error()
                if not done:
                    raise TimeoutError(
                        f"Timeout waiting for aggregation result for request_id='{self.request_id}' "
                        f"step {step_no} (received {len(step.inputs[0])}/{self.expected_workers})"
                    )
                step.reads += 1
                # Every worker that sent inputs reads the result once. Count against
                # those inputs: an Unregister may have lowered expected_workers since.
                if step.reads >= len(step.inputs[0]):
                    del self.steps[step_no]
                    self.last_completed_step = max(self.last_completed_step, step_no)
                return step.result
            except Exception as exc:
                # Fail the request before releasing the lock. Otherwise another
                # worker could complete a step this worker already gave up on.
                self.fail(exc)
                raise

    def compute_complete_steps(self, reduce_fn) -> None:
        """Compute every step that has inputs from all expected workers."""
        for step in self.steps.values():
            if step.result is None and len(step.inputs[0]) >= self.expected_workers:
                step.result = [
                    reduce_fn(aggregation_type, inputs)
                    for aggregation_type, inputs in zip(
                        step.aggregation_types, step.inputs
                    )
                ]
                self.condition.notify_all()

    def _add_inputs(
        self,
        step: _Step,
        step_no: int,
        aggregation_types: List[AggregationType],
        payloads: List[np.ndarray],
    ) -> None:
        if aggregation_types != step.aggregation_types:
            raise ValueError(
                f"Batch aggregation_type mismatch for request_id='{self.request_id}' "
                f"step {step_no} (expected {step.aggregation_types}, "
                f"got {aggregation_types})"
            )
        for idx, (aggregation_type, inputs, payload) in enumerate(
            zip(aggregation_types, step.inputs, payloads)
        ):
            if (
                aggregation_type != AggregationType.UNION
                and inputs
                and len(payload) != len(inputs[0])
            ):
                raise ValueError(
                    f"All vectors in batch op {idx} must have the same length "
                    f"(expected {len(inputs[0])}, got {len(payload)})"
                )
        for inputs, payload in zip(step.inputs, payloads):
            inputs.append(payload)


class AggregationServer(AggregationServerServicer):
    def __init__(self):
        self.aggregation_contexts: Dict[str, AggregationContext] = {}
        # request_id -> monotonic time of Cleanup (or stale eviction). Lets late
        # Aggregate/Unregister calls get ABORTED instead of looking unconfigured.
        self.cleaned_up_request_ids: Dict[str, float] = {}
        self.global_lock = threading.Lock()

    def _get_aggregation_context(self, request_id, context) -> AggregationContext:
        agg_ctx = self.aggregation_contexts.get(request_id)
        if agg_ctx:
            return agg_ctx
        with self.global_lock:
            agg_ctx = self.aggregation_contexts.get(request_id)
            if agg_ctx:
                return agg_ctx
            if request_id in self.cleaned_up_request_ids:
                msg = (
                    f"Aggregation for request_id='{request_id}' was cancelled "
                    "because the controller cleaned up the request."
                )
                logger.error(f"[AGGREGATE] {msg}")
                context.abort(grpc.StatusCode.ABORTED, msg)
            msg = f"request_id='{request_id}' is not configured."
            logger.error(msg)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
        return agg_ctx

    @staticmethod
    def _reduce(
        aggregation_type: AggregationType, vectors: list[np.ndarray]
    ) -> np.ndarray:
        if aggregation_type == AggregationType.UNION:
            return np.asarray(sorted(set(np.concatenate(vectors))), dtype=object)
        ufunc = REDUCERS[aggregation_type]
        dtype = _result_dtype(aggregation_type, [vector.dtype for vector in vectors])
        result = vectors[0].astype(dtype, copy=True)
        for vector in vectors[1:]:
            ufunc(result, vector.astype(dtype, copy=False), out=result)
        return result

    def _evict_stale_contexts(self) -> None:
        cutoff = time.monotonic() - STALE_CONTEXT_TTL_SECONDS
        stale_request_ids = [
            request_id
            for request_id, agg_ctx in self.aggregation_contexts.items()
            if agg_ctx.last_activity < cutoff
        ]
        for request_id in stale_request_ids:
            agg_ctx = self.aggregation_contexts.pop(request_id)
            self.cleaned_up_request_ids[request_id] = time.monotonic()
            agg_ctx.fail(
                AggregationCancelledError(
                    f"Aggregation for request_id='{request_id}' was cancelled "
                    "because its context was evicted as stale."
                ),
                log=False,
            )
            logger.warning(
                f"[CONFIGURE] Evicted stale AggregationContext for request_id='{request_id}' "
                f"(no activity for {STALE_CONTEXT_TTL_SECONDS}s and no Cleanup received)"
            )
        stale_cleaned = [
            request_id
            for request_id, cleaned_at in self.cleaned_up_request_ids.items()
            if cleaned_at < cutoff
        ]
        for request_id in stale_cleaned:
            del self.cleaned_up_request_ids[request_id]

    def Configure(self, request, context):
        with self.global_lock:
            self._evict_stale_contexts()
            if request.request_id in self.aggregation_contexts:
                logger.warning(
                    f"[CONFIGURE] Request context already exists for request_id='{request.request_id}'"
                )
                return ConfigureResponse(
                    status=Status.ALREADY_CONFIGURED,
                    status_message="Already configured for this request_id",
                )

            self.cleaned_up_request_ids.pop(request.request_id, None)
            self.aggregation_contexts[request.request_id] = AggregationContext(
                request.request_id, request.num_of_workers
            )

        logger.info(
            f"[CONFIGURE] Created AggregationContext for request_id='{request.request_id}' "
            f"with expected workers: {request.num_of_workers}"
        )
        return ConfigureResponse(status=Status.OK, status_message="Configured")

    def Unregister(self, request, context):
        agg_ctx = self._get_aggregation_context(request.request_id, context)
        agg_ctx.last_activity = time.monotonic()
        with agg_ctx.condition:
            if agg_ctx.expected_workers <= 1:
                msg = (
                    f"[UNREGISTER] request_id='{request.request_id}' cannot go below 1 worker "
                    f"(current: {agg_ctx.expected_workers})"
                )
                logger.warning(msg)
                return UnregisterResponse(
                    status=Status.MIN_WORKERS,
                    status_message="Minimum worker count reached",
                    remaining_workers=agg_ctx.expected_workers,
                )

            agg_ctx.expected_workers -= 1
            logger.info(
                f"[UNREGISTER] request_id='{request.request_id}' decreased expected workers "
                f"to {agg_ctx.expected_workers}"
            )

            try:
                agg_ctx.compute_complete_steps(self._reduce)
            except Exception as exc:  # noqa: BLE001
                agg_ctx.fail(exc)
                context.abort(_grpc_status(exc), str(exc))

        return UnregisterResponse(
            status=Status.OK,
            status_message="Unregistered one worker",
            remaining_workers=agg_ctx.expected_workers,
        )

    def Aggregate(self, request, context):
        agg_ctx = self._get_aggregation_context(request.request_id, context)
        agg_ctx.last_activity = time.monotonic()
        try:
            aggregation_types, payloads = _decode_operations(request)
            tensors = agg_ctx.aggregate(
                request.step, aggregation_types, payloads, self._reduce
            )
        except Exception as exc:  # noqa: BLE001
            agg_ctx.fail(exc)
            context.abort(_grpc_status(exc), str(exc))

        return AggregateResponse(
            tensors=[
                (
                    values_to_bytes(tensor)
                    if tensor.dtype == object
                    else ndarray_to_bytes(tensor)
                )
                for tensor in tensors
            ],
        )

    def Cleanup(self, request, context):
        with self.global_lock:
            agg_ctx = self.aggregation_contexts.pop(request.request_id, None)
            if agg_ctx is not None:
                self.cleaned_up_request_ids[request.request_id] = time.monotonic()

        if not agg_ctx:
            msg = f"[CLEANUP] No aggregation context found for request_id='{request.request_id}'"
            logger.warning(msg)
            return CleanupResponse(
                status=Status.NOT_FOUND, status_message="No such request_id"
            )

        # Fail the context even with no step in progress: an Aggregate call that
        # fetched it before the pop must not open a new step and wait out the
        # timeout. Log only when workers were actually cut off.
        with agg_ctx.condition:
            agg_ctx.fail(
                AggregationCancelledError(
                    f"Aggregation for request_id='{request.request_id}' was "
                    "cancelled because the controller cleaned up the request."
                ),
                log=bool(agg_ctx.steps),
            )

        logger.info(
            f"[CLEANUP] Removed AggregationContext for request_id='{request.request_id}'"
        )
        return CleanupResponse(status=Status.OK, status_message="Cleaned up")


def serve():
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_grpc_connections),
        options=grpc_message_options(config.max_message_mb),
    )
    add_AggregationServerServicer_to_server(AggregationServer(), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("Aggregation", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(f"0.0.0.0:{config.port}")
    server.start()
    logger.info(f"Aggregation server running on 0.0.0.0:{config.port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Aggregation server shutting down…")
        server.stop(0)


if __name__ == "__main__":
    serve()
