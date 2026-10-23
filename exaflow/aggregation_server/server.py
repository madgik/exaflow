import logging
import threading
import time
from concurrent import futures
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

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
"""
Aggregation server, threading model, and data flow
--------------------------------------------------

Overview
~~~~~~~~
Each `request_id` represents a round of vector aggregation across `expected_workers`
callers. Workers first register via `Configure(request_id, num_workers)` and then
call either:
- `Aggregate`: single operation (SUM / MIN / MAX) over a flat vector.
- `AggregateBatch`: multiple operations in one call; each operation defines its
  own aggregation type and vector.

Payloads are exchanged through the `tensor` bytes field. Numeric payloads use
Arrow tensor bytes, while `UNION` payloads use JSON-encoded bytes.

Synchronization model
~~~~~~~~~~~~~~~~~~~~~
For every `request_id` we keep an `AggregationContext` guarded by a single
`Condition`. The context follows a small state machine:
- `collecting`: accepting inputs. Once `expected_workers` inputs arrive we
  compute the result and move to `ready`.
- `ready` or `failed`: result (or error) is available. Each worker request that
  hits this state increments `acquired_count`. When all `expected_workers`
  have observed the result or error, the context is reset back to `collecting`.

All waits are bounded by `config.max_wait_for_aggregation_inputs` to avoid
deadlocks from missing workers.

`Cleanup` removes the context and cancels it: workers still waiting on it, for
a result or to start a step, get ABORTED at once instead of waiting for the
timeout. The controller calls `Cleanup` in a `finally` block, so when one worker
fails the others are released as soon as the controller sees the failure.
Late Aggregate/Unregister calls after Cleanup get ABORTED with a cleanup-specific
message, not INVALID_ARGUMENT (which is reserved for never-configured ids).

If `Cleanup` never arrives (for example the controller crashed), the context
would stay in memory with any inputs it collected. `Configure` therefore evicts
contexts with no `Aggregate` or `Unregister` call for
`STALE_CONTEXT_TTL_SECONDS`.

Error handling
~~~~~~~~~~~~~~
Any validation or computation error sets the context to `failed`, notifies
waiters, and counts the triggering request as consumed so the round can drain.
Subsequent workers immediately receive the same error instead of waiting
indefinitely.

`AggregationContext` raises plain Python exceptions and only the servicer
methods call `context.abort()`, mapping `ValueError` to INVALID_ARGUMENT,
`TimeoutError` to DEADLINE_EXCEEDED, `AggregationCancelledError` to ABORTED and
anything else to INTERNAL. gRPC's
`abort()` raises a bare `Exception`, so it must never run inside a
`try/except Exception` block or its status and details are lost.
"""


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


class AggregationCancelledError(Exception):
    """The request was cleaned up while a worker was still waiting on it."""


def _parse_aggregation_type(operation) -> AggregationType:
    try:
        return AggregationType(operation.aggregation_type)
    except ValueError:
        raise ValueError(
            f"Unsupported aggregation type '{operation.aggregation_type}'"
        ) from None


def _grpc_status(exc: Exception) -> grpc.StatusCode:
    if isinstance(exc, AggregationCancelledError):
        return grpc.StatusCode.ABORTED
    if isinstance(exc, TimeoutError):
        return grpc.StatusCode.DEADLINE_EXCEEDED
    if isinstance(exc, ValueError):
        return grpc.StatusCode.INVALID_ARGUMENT
    return grpc.StatusCode.INTERNAL


class AggregationState(Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    READY = "ready"
    FAILED = "failed"


class AggregationContext:
    def __init__(self, request_id: str, expected_workers: int):
        self.request_id = request_id
        self.expected_workers = expected_workers
        self.current_step = 0
        self.active_step: Optional[int] = None
        self.state = AggregationState.IDLE
        self.acquired_count = 0
        self.error: Optional[Exception] = None
        self.batch_ops: Optional[List[str]] = None
        self.batch_payloads: Optional[List[List[np.ndarray]]] = None
        self.batch_vector_lengths: Optional[List[Optional[int]]] = None
        self.batch_result: Optional[List[np.ndarray]] = None
        self.cancelled = False
        self.last_activity = time.monotonic()
        self.condition = threading.Condition()

    def reset_step(self) -> None:
        self.active_step = None
        self.state = AggregationState.IDLE
        self.acquired_count = 0
        self.error = None
        self.batch_ops = None
        self.batch_payloads = None
        self.batch_vector_lengths = None
        self.batch_result = None
        self.condition.notify_all()

    def fail(self, exc: Exception) -> None:
        self.error = exc
        self.state = AggregationState.FAILED
        self.condition.notify_all()

    def cancel(self) -> None:
        self.cancelled = True
        self.fail(
            AggregationCancelledError(
                f"Aggregation for request_id='{self.request_id}' was cancelled "
                "because the controller cleaned up the request."
            )
        )

    def ensure_step(self, step: int, operations) -> None:
        def _can_start_new_step() -> bool:
            return (
                self.active_step is None
                and self.state == AggregationState.IDLE
                and step == self.current_step + 1
            )

        while True:
            if self.cancelled:
                raise self.error
            if step <= self.current_step:
                raise ValueError(
                    f"Step {step} already completed for request_id='{self.request_id}'"
                )
            if self.active_step == step:
                self._validate_ops(operations)
                return
            if _can_start_new_step():
                self._start_step(step, operations)
                return

            notified = self.condition.wait(
                timeout=config.max_wait_for_aggregation_inputs
            )
            if not notified:
                raise TimeoutError(
                    f"Timeout waiting to start step {step} for request_id='{self.request_id}'"
                )

    def _start_step(self, step: int, operations) -> None:
        # Parsed first, so an unknown type fails the step before it starts.
        batch_ops = [_parse_aggregation_type(op) for op in operations]
        self.active_step = step
        self.state = AggregationState.COLLECTING
        self.acquired_count = 0
        self.error = None
        self.batch_ops = batch_ops
        self.batch_payloads = [[] for _ in operations]
        self.batch_vector_lengths = [None for _ in operations]
        self.batch_result = None

    def _validate_ops(self, operations) -> None:
        if self.batch_ops is None:
            raise RuntimeError("Batch ops not initialized for active step.")
        if len(operations) != len(self.batch_ops):
            raise ValueError(
                f"Batch op count mismatch for request_id='{self.request_id}' "
                f"(expected {len(self.batch_ops)}, got {len(operations)})"
            )
        for idx, op in enumerate(operations):
            if op.aggregation_type != self.batch_ops[idx]:
                raise ValueError(
                    f"Batch aggregation_type mismatch at index {idx} for request_id='{self.request_id}' "
                    f"(expected '{self.batch_ops[idx]}', got '{op.aggregation_type}')"
                )

    def store_payloads(self, request, decode_fn) -> None:
        assert self.batch_payloads is not None
        assert self.batch_vector_lengths is not None
        for idx, op in enumerate(request.operations):
            payload = decode_fn(op.tensor, op.aggregation_type, request.request_id)
            vector_length = len(payload)
            if op.aggregation_type != AggregationType.UNION:
                if payload.dtype.kind not in NUMERIC_KINDS:
                    raise ValueError(
                        "Numeric aggregation requires numeric tensor payloads."
                    )
                expected_len = self.batch_vector_lengths[idx]
                if expected_len is None:
                    self.batch_vector_lengths[idx] = vector_length
                elif vector_length != expected_len:
                    raise ValueError(
                        f"All vectors in batch op {idx} must have the same length "
                        f"(expected {expected_len}, got {vector_length})"
                    )
            self.batch_payloads[idx].append(payload)

    def compute(self, reduce_fn) -> None:
        assert self.batch_ops is not None and self.batch_payloads is not None
        self.batch_result = [
            reduce_fn(op_type, payloads)
            for op_type, payloads in zip(self.batch_ops, self.batch_payloads)
        ]
        self.state = AggregationState.READY
        self.condition.notify_all()

    def wait_for_ready(self) -> None:
        ready = self.condition.wait_for(
            lambda: self.state in {AggregationState.READY, AggregationState.FAILED},
            timeout=config.max_wait_for_aggregation_inputs,
        )
        if ready:
            return
        received = len(self.batch_payloads[0]) if self.batch_payloads else 0
        msg = (
            f"Timeout waiting for aggregation result for request_id='{self.request_id}' "
            f"(received {received}/{self.expected_workers})"
        )
        logger.error(f"[AGGREGATE] {msg}")
        timeout_exc = TimeoutError(msg)
        self.fail(timeout_exc)
        raise timeout_exc

    def consume(self):
        if self.cancelled:
            raise self.error
        if self.error is not None:
            error = self.error
            self._finish_step(reset_step=False)
            logger.error(f"[AGGREGATE] Error during batch aggregation: {error}")
            raise error

        if not self.batch_result:
            raise RuntimeError("Batch result missing.")

        tensors = self.batch_result
        self._finish_step(reset_step=True)
        return tensors

    def _finish_step(self, reset_step: bool) -> None:
        self.acquired_count += 1
        if self.acquired_count >= self.expected_workers:
            self.current_step = self.active_step or self.current_step
            if reset_step:
                self.reset_step()
            else:
                self.state = AggregationState.IDLE
                self.condition.notify_all()


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

    def _decode_vector(
        self, tensor: bytes, aggregation_type: str, request_id: str
    ) -> np.ndarray:
        if tensor:
            if aggregation_type == AggregationType.UNION:
                return bytes_to_values(tensor)
            return bytes_to_ndarray(tensor)
        raise ValueError(
            f"[AGGREGATE] request_id='{request_id}' missing tensor payload"
        )

    @staticmethod
    def _reduce(
        aggregation_type: AggregationType, vectors: list[np.ndarray]
    ) -> np.ndarray:
        if aggregation_type == AggregationType.UNION:
            return np.asarray(sorted(set(np.concatenate(vectors))), dtype=object)
        ufunc = REDUCERS.get(aggregation_type)
        if ufunc is None:
            raise ValueError(f"Unsupported aggregation type '{aggregation_type}'")
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
            with agg_ctx.condition:
                agg_ctx.cancel()
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

            if (
                agg_ctx.state == AggregationState.COLLECTING
                and agg_ctx.batch_payloads
                and len(agg_ctx.batch_payloads[0]) >= agg_ctx.expected_workers
            ):
                try:
                    agg_ctx.compute(self._reduce)
                except Exception as exc:  # noqa: BLE001
                    agg_ctx.fail(exc)
                    logger.error(f"[UNREGISTER/AGGREGATE] {exc}", exc_info=exc)
                    context.abort(_grpc_status(exc), str(exc))

            agg_ctx.condition.notify_all()

        return UnregisterResponse(
            status=Status.OK,
            status_message="Unregistered one worker",
            remaining_workers=agg_ctx.expected_workers,
        )

    def Aggregate(self, request, context):
        agg_ctx = self._get_aggregation_context(request.request_id, context)
        agg_ctx.last_activity = time.monotonic()

        with agg_ctx.condition:
            try:
                agg_ctx.ensure_step(request.step, request.operations)
                agg_ctx.store_payloads(
                    request,
                    lambda tensor, aggregation_type, request_id=request.request_id: (
                        self._decode_vector(tensor, aggregation_type, request_id)
                    ),
                )
                received_workers = (
                    len(agg_ctx.batch_payloads[0]) if agg_ctx.batch_payloads else 0
                )
                logger.info(
                    "[AGGREGATE] Received worker results for request_id='%s' "
                    "(%d/%d workers reported)",
                    request.request_id,
                    received_workers,
                    agg_ctx.expected_workers,
                )

                if (
                    agg_ctx.state == AggregationState.COLLECTING
                    and agg_ctx.batch_payloads
                    and len(agg_ctx.batch_payloads[0]) >= agg_ctx.expected_workers
                ):
                    agg_ctx.compute(self._reduce)
            except Exception as exc:  # noqa: BLE001
                agg_ctx.fail(exc)
                logger.error(f"[AGGREGATE] {exc}", exc_info=exc)
                context.abort(_grpc_status(exc), str(exc))

            try:
                agg_ctx.wait_for_ready()
                tensors = agg_ctx.consume()
            except Exception as exc:  # noqa: BLE001
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

        with agg_ctx.condition:
            agg_ctx.cancel()

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
