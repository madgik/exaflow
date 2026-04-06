import logging
import threading
from concurrent import futures
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import grpc
import numpy as np
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from exaflow.aggregation_server import config
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

from .constants import AggregationType
from .serialization import bytes_to_ndarray
from .serialization import bytes_to_values
from .serialization import ndarray_to_bytes
from .serialization import values_to_bytes

logger = logging.getLogger("AggregationServer")
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

Payloads can arrive either as legacy repeated doubles (`vectors`) or as Arrow
tensors (`tensor`). Responses always include both a Python list and the binary
tensor form for efficiency.

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

Error handling
~~~~~~~~~~~~~~
Any validation or computation error sets the context to `failed`, notifies
waiters, and counts the triggering request as consumed so the round can drain.
Subsequent workers immediately receive the same error instead of waiting
indefinitely.
"""


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
        self.batch_vectors: Optional[List[List[np.ndarray]]] = None
        self.batch_vector_lengths: Optional[List[Optional[int]]] = None
        self.batch_result: Optional[Tuple[List[float], List[int], List[np.ndarray]]] = (
            None
        )
        self.condition = threading.Condition()

    def reset_step(self) -> None:
        self.active_step = None
        self.state = AggregationState.IDLE
        self.acquired_count = 0
        self.error = None
        self.batch_ops = None
        self.batch_vectors = None
        self.batch_vector_lengths = None
        self.batch_result = None
        self.condition.notify_all()

    def fail(self, exc: Exception) -> None:
        self.error = exc
        self.state = AggregationState.FAILED
        self.condition.notify_all()

    def ensure_step(self, step: int, operations, context) -> None:
        def _can_start_new_step() -> bool:
            return (
                self.active_step is None
                and self.state == AggregationState.IDLE
                and step == self.current_step + 1
            )

        while True:
            if step <= self.current_step:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Step {step} already completed for request_id='{self.request_id}'",
                )
            if self.active_step == step:
                self._validate_ops(operations, context)
                return
            if _can_start_new_step():
                self._start_step(step, operations)
                return

            notified = self.condition.wait(
                timeout=config.max_wait_for_aggregation_inputs
            )
            if not notified:
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    f"Timeout waiting to start step {step} for request_id='{self.request_id}'",
                )

    def _start_step(self, step: int, operations) -> None:
        self.active_step = step
        self.state = AggregationState.COLLECTING
        self.acquired_count = 0
        self.error = None
        self.batch_ops = [op.aggregation_type for op in operations]
        self.batch_vectors = [[] for _ in operations]
        self.batch_vector_lengths = [None for _ in operations]
        self.batch_result = None

    def _validate_ops(self, operations, context) -> None:
        if self.batch_ops is None:
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Batch ops not initialized for active step.",
            )
        if len(operations) != len(self.batch_ops):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Batch op count mismatch for request_id='{self.request_id}' "
                f"(expected {len(self.batch_ops)}, got {len(operations)})",
            )
        for idx, op in enumerate(operations):
            if op.aggregation_type != self.batch_ops[idx]:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Batch aggregation_type mismatch at index {idx} for request_id='{self.request_id}' "
                    f"(expected '{self.batch_ops[idx]}', got '{op.aggregation_type}')",
                )

    def store_vectors(self, request, decode_fn) -> int:
        assert self.batch_vectors is not None
        assert self.batch_vector_lengths is not None
        for idx, op in enumerate(request.operations):
            vector = decode_fn(
                op.tensor, op.vectors, op.aggregation_type, request.request_id
            )
            vector_length = len(vector)
            if op.aggregation_type != AggregationType.UNION.value:
                expected_len = self.batch_vector_lengths[idx]
                if expected_len is None:
                    self.batch_vector_lengths[idx] = vector_length
                elif vector_length != expected_len:
                    raise ValueError(
                        f"All vectors in batch op {idx} must have the same length "
                        f"(expected {expected_len}, got {vector_length})"
                    )
            self.batch_vectors[idx].append(vector)

        return len(self.batch_vectors[0])

    def compute(self, aggregation_fn) -> None:
        assert self.batch_ops is not None and self.batch_vectors is not None
        flat_results: List[float] = []
        offsets = [0]
        tensor_results: List[np.ndarray] = []
        for idx, op_type in enumerate(self.batch_ops):
            agg_fn = aggregation_fn(op_type)
            vectors = self.batch_vectors[idx]
            res = agg_fn(vectors)
            tensor_results.append(res)
            if op_type == AggregationType.UNION.value:
                offsets.append(len(flat_results))
            else:
                flat_results.extend(res.tolist())
                offsets.append(len(flat_results))

        self.batch_result = (flat_results, offsets, tensor_results)
        self.state = AggregationState.READY
        self.condition.notify_all()

    def wait_for_ready(self, context) -> None:
        ready = self.condition.wait_for(
            lambda: self.state in {AggregationState.READY, AggregationState.FAILED},
            timeout=config.max_wait_for_aggregation_inputs,
        )
        if ready:
            return
        received = len(self.batch_vectors[0]) if self.batch_vectors else 0
        msg = (
            f"Timeout waiting for aggregation result for request_id='{self.request_id}' "
            f"(received {received}/{self.expected_workers})"
        )
        logger.error(f"[AGGREGATE] {msg}")
        timeout_exc = TimeoutError(msg)
        self.fail(timeout_exc)
        context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, msg)

    def consume(self, context):
        if self.error is not None:
            error = self.error
            status = AggregationServer._status_for_exception_static(error)
            self._finish_step(reset_step=False)
            logger.error(f"[AGGREGATE] Error during batch aggregation: {error}")
            context.abort(status, str(error))

        if not self.batch_result:
            context.abort(grpc.StatusCode.INTERNAL, "Batch result missing.")

        results, offsets, tensors = self.batch_result
        self._finish_step(reset_step=True)
        return results, offsets, tensors

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
        self.global_lock = threading.Lock()

    @staticmethod
    def _status_for_exception_static(exc: Exception) -> grpc.StatusCode:
        if isinstance(exc, grpc.RpcError):
            try:
                return exc.code()
            except Exception:  # pragma: no cover
                pass
        if isinstance(exc, TimeoutError):
            return grpc.StatusCode.DEADLINE_EXCEEDED
        if isinstance(exc, ValueError):
            return grpc.StatusCode.INVALID_ARGUMENT
        return grpc.StatusCode.INTERNAL

    def _status_for_exception(self, exc: Exception) -> grpc.StatusCode:
        return AggregationServer._status_for_exception_static(exc)

    def _get_aggregation_context(self, request_id, context) -> AggregationContext:
        agg_ctx = self.aggregation_contexts.get(request_id)
        if not agg_ctx:
            msg = (
                f"[CONFIGURE] request_id='{request_id}' is not configured "
                "or has already been cleaned up."
            )
            logger.error(msg)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
        return agg_ctx

    def _decode_vector(
        self, tensor: bytes, vectors, aggregation_type: str, request_id: str
    ) -> np.ndarray:
        if tensor:
            if aggregation_type == AggregationType.UNION.value:
                return bytes_to_values(tensor)
            return bytes_to_ndarray(tensor)
        if vectors:
            return np.asarray(vectors, dtype=np.float64)
        raise ValueError(
            f"[AGGREGATE] request_id='{request_id}' missing tensor or vector payload"
        )

    def _aggregation_fn(self, aggregation_type: str):
        try:
            agg_type = AggregationType(aggregation_type)
        except ValueError as exc:  # noqa: PERF203
            raise ValueError(
                f"Unsupported aggregation type '{aggregation_type}'"
            ) from exc

        if agg_type == AggregationType.SUM:
            return lambda vectors: np.sum(vectors, axis=0)
        if agg_type == AggregationType.MIN:
            return lambda vectors: np.min(vectors, axis=0)
        if agg_type == AggregationType.MAX:
            return lambda vectors: np.max(vectors, axis=0)
        if agg_type == AggregationType.UNION:
            return lambda vectors: np.asarray(
                sorted(set(np.concatenate(vectors))), dtype=object
            )

        raise ValueError(f"Unhandled aggregation type '{aggregation_type}'")

    def Configure(self, request, context):
        with self.global_lock:
            if request.request_id in self.aggregation_contexts:
                logger.warning(
                    f"[CONFIGURE] Request context already exists for request_id='{request.request_id}'"
                )
                return ConfigureResponse(
                    status=Status.ALREADY_CONFIGURED,
                    status_message="Already configured for this request_id",
                )

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
                and agg_ctx.batch_vectors
                and len(agg_ctx.batch_vectors[0]) >= agg_ctx.expected_workers
            ):
                try:
                    agg_ctx.compute(self._aggregation_fn)
                except Exception as exc:  # noqa: BLE001
                    agg_ctx.fail(exc)
                    status = self._status_for_exception(exc)
                    logger.error(f"[UNREGISTER/AGGREGATE] {exc}", exc_info=exc)
                    context.abort(status, str(exc))

            agg_ctx.condition.notify_all()

        return UnregisterResponse(
            status=Status.OK,
            status_message="Unregistered one worker",
            remaining_workers=agg_ctx.expected_workers,
        )

    def Aggregate(self, request, context):
        agg_ctx = self._get_aggregation_context(request.request_id, context)

        with agg_ctx.condition:
            try:
                agg_ctx.ensure_step(request.step, request.operations, context)
                agg_ctx.store_vectors(
                    request,
                    lambda tensor, vectors, aggregation_type, request_id=request.request_id: (
                        self._decode_vector(
                            tensor, vectors, aggregation_type, request_id
                        )
                    ),
                )
                received_workers = (
                    len(agg_ctx.batch_vectors[0]) if agg_ctx.batch_vectors else 0
                )
                logger.info(
                    "[AGGREGATE] Received worker results for context_id='%s' "
                    "(%d/%d workers reported)",
                    request.request_id,
                    received_workers,
                    agg_ctx.expected_workers,
                )

                if (
                    agg_ctx.state == AggregationState.COLLECTING
                    and agg_ctx.batch_vectors
                    and len(agg_ctx.batch_vectors[0]) >= agg_ctx.expected_workers
                ):
                    agg_ctx.compute(self._aggregation_fn)
            except grpc.RpcError as exc:
                agg_ctx.fail(exc)
                logger.error(f"[AGGREGATE] {exc}", exc_info=exc)
                raise
            except Exception as exc:  # noqa: BLE001
                agg_ctx.fail(exc)
                status = self._status_for_exception(exc)
                logger.error(f"[AGGREGATE] {exc}", exc_info=exc)
                context.abort(status, str(exc))

            agg_ctx.wait_for_ready(context)
            results, offsets, tensors = agg_ctx.consume(context)

        return AggregateResponse(
            results=results,
            offsets=offsets,
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

        if not agg_ctx:
            msg = f"[CLEANUP] No aggregation context found for request_id='{request.request_id}'"
            logger.warning(msg)
            return CleanupResponse(
                status=Status.NOT_FOUND, status_message="No such request_id"
            )

        with agg_ctx.condition:
            agg_ctx.reset_step()

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
        futures.ThreadPoolExecutor(max_workers=config.max_grpc_connections)
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
