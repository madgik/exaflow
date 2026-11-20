"""gRPC aggregation server: combines vectors from the workers of one request.

Lifecycle of a request
----------------------
- The controller calls `Configure(request_id, context_id, num_of_workers)` before running
  the algorithm and `Cleanup(request_id, context_id)` when the request ends, in a `finally`
  block. Both calls carry an ownership token in gRPC metadata. Cleanup deletes
  the context only when the token matches the one stored at Configure, so a
  duplicate request that lost its response cannot remove someone else's context.
- Each worker calls `Aggregate(request_id, context_id, step, operations)` from its UDF.
  `step` is numbered by the worker's client (1, 2, 3, ...). `operations` holds
  one or more `(aggregation_type, tensor)` pairs. Numeric types (`SUM`, `MIN`,
  `MAX`) send Arrow tensors; `UNION` sends JSON-encoded values.
- A worker that drops out (too little data, bad input) calls `Unregister`, which
  lowers the number of inputs each later step waits for.

Synchronization
---------------
Each execution attempt has an `AggregationContext` guarded by one `Condition`. It keeps
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
for never-configured ids). A `Cleanup` that arrives before `Configure` is
still recorded. `Configure` for that id returns ERROR until the record expires
or the cancelled-id set drops it, so it does not create a context that lives
for `STALE_CONTEXT_TTL_SECONDS`. If `Cleanup` never arrives (for example the
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
from dataclasses import field
from typing import Collection
from typing import Dict
from typing import List
from typing import Literal
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
from exaflow.grpc_tls import DuplicateWorkerSubmissionError
from exaflow.grpc_tls import PeerIdentity
from exaflow.grpc_tls import UnauthenticatedError
from exaflow.grpc_tls import UnauthorizedError
from exaflow.grpc_tls import build_server_credentials
from exaflow.grpc_tls import extract_peer_identity
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
# Cleanup is remembered even when it arrives before Configure. The set is
# capped so a stream of unknown ids cannot grow it without bound; the TTL
# still drops entries that age out first.
CANCELLED_CONTEXT_ID_LIMIT = 1024


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
        raise ValueError(
            f"request_id='{request.request_id}' context_id='{request.context_id}' sent no operations"
        )
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
                f"request_id='{request.request_id}' context_id='{request.context_id}' missing tensor payload"
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
    # With mTLS, one submission per certificate-bound worker id per step.
    submitted_workers: set[str] = field(default_factory=set)


# gRPC metadata key. The controller sends a fresh token on Configure and the
# same token on Cleanup. Workers do not send it; the context stays keyed by
# request id so their Aggregate calls are unchanged.
OWNERSHIP_TOKEN_METADATA_KEY = "x-aggregation-owner"


def _ownership_token(context) -> str:
    invocation = getattr(context, "invocation_metadata", None)
    if not callable(invocation):
        return ""
    for item in invocation() or ():
        key, value = item[0], item[1]
        if str(key).lower() == OWNERSHIP_TOKEN_METADATA_KEY:
            return str(value)
    return ""


class AggregationContext:
    """Aggregation state of one request, guarded by `condition`."""

    def __init__(
        self,
        request_id: str,
        expected_workers: int,
        ownership_token: str = "",
        context_id: str = "",
        worker_ids: Collection[str] | None = None,
    ):
        self.request_id = request_id
        self.context_id = context_id
        self.expected_workers = expected_workers
        self.ownership_token = ownership_token
        # None means no certificate-bound membership check (plaintext mode).
        # A set means only these worker ids may submit, at most once per step.
        self.active_worker_ids: Optional[set[str]] = (
            set(worker_ids) if worker_ids is not None else None
        )
        self.configured_worker_ids: Optional[frozenset[str]] = (
            frozenset(worker_ids) if worker_ids is not None else None
        )
        if self.active_worker_ids is not None:
            self.expected_workers = len(self.active_worker_ids)
        self.steps: Dict[int, _Step] = {}
        self.last_completed_step = 0
        self.error: Optional[Exception] = None
        self.last_activity = time.monotonic()
        self.condition = threading.Condition()

    def check_worker_step_admission(
        self, worker_id: Optional[str], step_no: int
    ) -> None:
        """Reject non-members and duplicate/replayed steps without failing the context."""
        with self.condition:
            if self.active_worker_ids is None:
                return
            if worker_id is None or worker_id not in self.active_worker_ids:
                raise UnauthorizedError(
                    f"worker '{worker_id}' is not a member of request_id='{self.request_id}' "
                    f"context_id='{self.context_id}'"
                )
            if step_no <= self.last_completed_step:
                raise DuplicateWorkerSubmissionError(
                    f"worker '{worker_id}' already completed step {step_no} for "
                    f"request_id='{self.request_id}' context_id='{self.context_id}'"
                )
            existing_step = self.steps.get(step_no)
            if (
                existing_step is not None
                and worker_id in existing_step.submitted_workers
            ):
                raise DuplicateWorkerSubmissionError(
                    f"worker '{worker_id}' already submitted step {step_no} for "
                    f"request_id='{self.request_id}' context_id='{self.context_id}'"
                )

    def fail(self, exc: Exception, *, log: bool = True) -> None:
        """End the request: every waiting and later call raises the first error."""
        with self.condition:
            if self.error is None:
                self.error = exc
                if log:
                    logger.error(
                        f"[AGGREGATE] request_id='{self.request_id}' context_id='{self.context_id}' failed: {exc}",
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
        worker_id: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Add one worker's inputs for `step_no` and wait for the combined result."""
        with self.condition:
            # Authorization and duplicate-submission failures are local to the
            # caller: never call self.fail(), or an untrusted/duplicate worker
            # could poison an otherwise healthy aggregation.
            self.check_worker_step_admission(worker_id, step_no)
            try:
                if self.error is not None:
                    self._raise_error()
                if step_no <= self.last_completed_step:
                    raise ValueError(
                        f"Step {step_no} already completed for request_id='{self.request_id}' context_id='{self.context_id}'"
                    )
                step = self.steps.setdefault(
                    step_no, _Step(aggregation_types, [[] for _ in aggregation_types])
                )
                if step.result is not None:
                    raise ValueError(
                        f"Step {step_no} already completed for request_id='{self.request_id}' context_id='{self.context_id}'"
                    )
                if self.active_worker_ids is not None:
                    step.submitted_workers.add(worker_id)
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
                        f"Timeout waiting for aggregation result for request_id='{self.request_id}' context_id='{self.context_id}' "
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
            if step.result is not None:
                continue
            ready = (
                self.active_worker_ids <= step.submitted_workers
                if self.active_worker_ids is not None
                else len(step.inputs[0]) >= self.expected_workers
            )
            if ready:
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
                f"Batch aggregation_type mismatch for request_id='{self.request_id}' context_id='{self.context_id}' "
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
    def __init__(
        self,
        *,
        require_mtls: bool = False,
        expected_federation: str | None = None,
    ):
        self.require_mtls = require_mtls
        self.expected_federation = expected_federation
        self.aggregation_contexts: Dict[str, AggregationContext] = {}
        # context_id -> monotonic time of Cleanup (or stale eviction). Lets late
        # Aggregate/Unregister calls get ABORTED instead of looking unconfigured.
        self.cleaned_up_context_ids: Dict[str, float] = {}
        self.global_lock = threading.Lock()

    def _authorize(
        self, context, required_role: Literal["controller", "worker"]
    ) -> Optional[PeerIdentity]:
        """Authenticate the mTLS peer and check its role for this RPC.

        Must be called before touching aggregation state and outside any
        ``try/except Exception``: ``context.abort()`` raises a bare Exception,
        so catching it would lose the gRPC status.
        """
        if not self.require_mtls:
            return None
        try:
            identity = extract_peer_identity(
                context, expected_federation=self.expected_federation
            )
        except UnauthenticatedError as exc:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))
            raise AssertionError("context.abort() did not raise") from exc
        except UnauthorizedError as exc:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise AssertionError("context.abort() did not raise") from exc
        if identity.role != required_role:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                f"RPC requires role '{required_role}', got '{identity.role}'",
            )
            raise AssertionError("context.abort() did not raise")
        return identity

    @staticmethod
    def _validate_context_request(agg_ctx, request_id, context):
        if agg_ctx.request_id != request_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "request_id does not match the configured context_id",
            )
        return agg_ctx

    def _get_aggregation_context(
        self, context_id, context, request_id
    ) -> AggregationContext:
        if not context_id or not request_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "request_id and context_id are required",
            )
        agg_ctx = self.aggregation_contexts.get(context_id)
        if agg_ctx:
            return self._validate_context_request(agg_ctx, request_id, context)
        with self.global_lock:
            agg_ctx = self.aggregation_contexts.get(context_id)
            if agg_ctx:
                return self._validate_context_request(agg_ctx, request_id, context)
            label = f"request_id='{request_id}' context_id='{context_id}'"
            if context_id in self.cleaned_up_context_ids:
                msg = f"Aggregation for {label} was cancelled because the controller cleaned up the request."
                logger.error("[AGGREGATE] %s", msg)
                context.abort(grpc.StatusCode.ABORTED, msg)
            msg = f"{label} is not configured."
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
        stale_context_ids = [
            context_id
            for context_id, agg_ctx in self.aggregation_contexts.items()
            if agg_ctx.last_activity < cutoff
        ]
        for context_id in stale_context_ids:
            agg_ctx = self.aggregation_contexts.pop(context_id)
            self.cleaned_up_context_ids[context_id] = time.monotonic()
            agg_ctx.fail(
                AggregationCancelledError(
                    f"Aggregation for request_id='{agg_ctx.request_id}' context_id='{context_id}' was cancelled "
                    "because its context was evicted as stale."
                ),
                log=False,
            )
            logger.warning(
                f"[CONFIGURE] Evicted stale AggregationContext for request_id='{agg_ctx.request_id}' context_id='{context_id}' "
                f"(no activity for {STALE_CONTEXT_TTL_SECONDS}s and no Cleanup received)"
            )
        stale_cleaned = [
            context_id
            for context_id, cleaned_at in self.cleaned_up_context_ids.items()
            if cleaned_at < cutoff
        ]
        for context_id in stale_cleaned:
            del self.cleaned_up_context_ids[context_id]

    def _remember_cancelled(self, context_id: str) -> None:
        """Record a Cleanup, including one that arrived before Configure."""
        self.cleaned_up_context_ids[context_id] = time.monotonic()
        self._evict_stale_contexts()
        overflow = len(self.cleaned_up_context_ids) - CANCELLED_CONTEXT_ID_LIMIT
        if overflow <= 0:
            return
        oldest = sorted(
            self.cleaned_up_context_ids,
            key=self.cleaned_up_context_ids.__getitem__,
        )
        for stale_id in oldest[:overflow]:
            del self.cleaned_up_context_ids[stale_id]

    def Configure(self, request, context):
        self._authorize(context, "controller")
        if not request.context_id or not request.request_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "request_id and context_id are required",
            )
        worker_ids = list(request.worker_ids)
        if any(not str(worker_id).strip() for worker_id in worker_ids):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "worker_ids must not contain blank values",
            )
        if len(set(worker_ids)) != len(worker_ids):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "worker_ids must not contain duplicates",
            )
        if request.num_of_workers < 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "num_of_workers must be positive",
            )
        if (
            worker_ids
            and request.num_of_workers != 0
            and len(worker_ids) != request.num_of_workers
        ):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "worker_ids count does not match num_of_workers",
            )
        if self.require_mtls and not worker_ids:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "worker_ids are required when mTLS authentication is enabled",
            )
        expected_workers = len(worker_ids) if worker_ids else request.num_of_workers
        if expected_workers < 1:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "num_of_workers must be at least 1",
            )
        with self.global_lock:
            self._evict_stale_contexts()
            current = self.aggregation_contexts.get(request.context_id)
            if current is not None:
                self._validate_context_request(current, request.request_id, context)
                if (
                    self.require_mtls
                    and current.configured_worker_ids is not None
                    and frozenset(worker_ids) != current.configured_worker_ids
                ):
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "worker_ids do not match the configured context_id",
                    )
                logger.warning(
                    f"[CONFIGURE] Request context already exists for request_id='{request.request_id}' context_id='{request.context_id}'"
                )
                return ConfigureResponse(
                    status=Status.ALREADY_CONFIGURED,
                    status_message="Already configured for this context_id",
                )

            if request.context_id in self.cleaned_up_context_ids:
                logger.warning(
                    "[CONFIGURE] Refusing request_id='%s': cleanup already cancelled it",
                    f"{request.request_id} context_id={request.context_id}",
                )
                return ConfigureResponse(
                    status=Status.ERROR,
                    status_message="Request was already cancelled",
                )
            self.aggregation_contexts[request.context_id] = AggregationContext(
                request.request_id,
                expected_workers,
                _ownership_token(context),
                context_id=request.context_id,
                worker_ids=worker_ids if self.require_mtls else None,
            )

        logger.info(
            f"[CONFIGURE] Created AggregationContext for request_id='{request.request_id}' context_id='{request.context_id}' "
            f"with expected workers: {expected_workers}"
        )
        return ConfigureResponse(status=Status.OK, status_message="Configured")

    def Unregister(self, request, context):
        identity = self._authorize(context, "worker")
        agg_ctx = self._get_aggregation_context(
            request.context_id, context, request.request_id
        )
        worker_id = identity.worker_id if identity else None
        with agg_ctx.condition:
            if agg_ctx.active_worker_ids is not None:
                # A worker outside the allow-list (including one that already
                # unregistered) must not shrink the expected worker count.
                if worker_id is None or worker_id not in agg_ctx.active_worker_ids:
                    context.abort(
                        grpc.StatusCode.PERMISSION_DENIED,
                        f"worker '{worker_id}' is not a member of request_id='{request.request_id}' "
                        f"context_id='{request.context_id}'",
                    )
                if agg_ctx.error is not None:
                    context.abort(_grpc_status(agg_ctx.error), str(agg_ctx.error))
                if any(
                    worker_id in step.submitted_workers
                    for step in agg_ctx.steps.values()
                ):
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        f"worker '{worker_id}' cannot unregister after submitting to an in-progress step",
                    )
                agg_ctx.last_activity = time.monotonic()
                if len(agg_ctx.active_worker_ids) <= 1:
                    msg = (
                        f"[UNREGISTER] request_id='{request.request_id}' context_id='{request.context_id}' cannot go below 1 worker "
                        f"(current: {agg_ctx.expected_workers})"
                    )
                    logger.warning(msg)
                    return UnregisterResponse(
                        status=Status.MIN_WORKERS,
                        status_message="Minimum worker count reached",
                        remaining_workers=agg_ctx.expected_workers,
                    )
                agg_ctx.active_worker_ids.remove(worker_id)
                agg_ctx.expected_workers = len(agg_ctx.active_worker_ids)
                logger.info(
                    f"[UNREGISTER] request_id='{request.request_id}' context_id='{request.context_id}' removed worker '{worker_id}'; "
                    f"expected workers is now {agg_ctx.expected_workers}"
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

            if agg_ctx.error is not None:
                context.abort(_grpc_status(agg_ctx.error), str(agg_ctx.error))
            agg_ctx.last_activity = time.monotonic()
            if agg_ctx.expected_workers <= 1:
                msg = (
                    f"[UNREGISTER] request_id='{request.request_id}' context_id='{request.context_id}' cannot go below 1 worker "
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
                f"[UNREGISTER] request_id='{request.request_id}' context_id='{request.context_id}' decreased expected workers "
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
        identity = self._authorize(context, "worker")
        agg_ctx = self._get_aggregation_context(
            request.context_id, context, request.request_id
        )
        worker_id = identity.worker_id if identity else None
        try:
            agg_ctx.check_worker_step_admission(worker_id, request.step)
        except UnauthorizedError as exc:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        except DuplicateWorkerSubmissionError as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        agg_ctx.last_activity = time.monotonic()
        try:
            aggregation_types, payloads = _decode_operations(request)
            tensors = agg_ctx.aggregate(
                request.step,
                aggregation_types,
                payloads,
                self._reduce,
                worker_id=worker_id,
            )
        except UnauthorizedError as exc:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        except DuplicateWorkerSubmissionError as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
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
        self._authorize(context, "controller")
        if not request.context_id or not request.request_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "request_id and context_id are required",
            )
        token = _ownership_token(context)
        with self.global_lock:
            current = self.aggregation_contexts.get(request.context_id)
            if current is not None:
                self._validate_context_request(current, request.request_id, context)
            if current is not None and current.ownership_token != token:
                logger.warning(
                    "[CLEANUP] Refused to remove request_id='%s': the ownership "
                    "token does not match the caller that configured it",
                    f"{request.request_id} context_id={request.context_id}",
                )
                return CleanupResponse(
                    status=Status.ERROR,
                    status_message="Aggregation context is owned by another caller",
                )
            agg_ctx = self.aggregation_contexts.pop(request.context_id, None)
            self._remember_cancelled(request.context_id)

        if not agg_ctx:
            msg = f"[CLEANUP] No aggregation context found for request_id='{request.request_id}' context_id='{request.context_id}'"
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
                    f"Aggregation for request_id='{request.request_id}' context_id='{request.context_id}' was "
                    "cancelled because the controller cleaned up the request."
                ),
                log=bool(agg_ctx.steps),
            )

        logger.info(
            f"[CLEANUP] Removed AggregationContext for request_id='{request.request_id}' context_id='{request.context_id}'"
        )
        return CleanupResponse(status=Status.OK, status_message="Cleaned up")


def _create_servers():
    """Create the main server and the optional plaintext health server.

    The dedicated health server exists so kubelet's gRPC probe can reach
    ``grpc.health.v1.Health`` without TLS while the aggregation RPCs stay
    behind mTLS. It registers the health servicer only; ``AggregationServer``
    is never exposed on that port.
    """
    if config.tls.enabled:
        if not config.tls.require_client_auth:
            raise ValueError(
                "Aggregation server TLS requires client authentication "
                "(require_client_auth must be true)"
            )
        if not (config.federation or "").strip():
            raise ValueError(
                "FEDERATION must be set when aggregation server mTLS is enabled"
            )
    if config.health_port > 0 and config.health_port == config.port:
        raise ValueError(
            f"health_port ({config.health_port}) must differ from port ({config.port})"
        )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_grpc_connections),
        options=grpc_message_options(config.max_message_mb),
    )
    add_AggregationServerServicer_to_server(
        AggregationServer(
            require_mtls=config.tls.enabled,
            expected_federation=(config.federation or "").strip() or None,
        ),
        server,
    )

    health_servicer = health.HealthServicer()
    health_servicer.set("Aggregation", health_pb2.HealthCheckResponse.SERVING)
    health_server = None
    if config.health_port > 0:
        health_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=2),
            options=grpc_message_options(config.max_message_mb),
        )
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, health_server)
    if not config.tls.enabled or config.health_port <= 0:
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    return server, health_server


def serve():
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server, health_server = _create_servers()
    listen_addr = f"0.0.0.0:{config.port}"
    if config.tls.enabled:
        bound_port = server.add_secure_port(
            listen_addr, build_server_credentials(config.tls)
        )
    else:
        bound_port = server.add_insecure_port(listen_addr)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind aggregation server on {listen_addr}")
    server.start()
    logger.info(f"Aggregation server running on {listen_addr}")

    if health_server is not None:
        health_addr = f"0.0.0.0:{config.health_port}"
        bound_health_port = health_server.add_insecure_port(health_addr)
        if bound_health_port == 0:
            server.stop(0)
            raise RuntimeError(
                f"Failed to bind aggregation health server on {health_addr}"
            )
        health_server.start()
        logger.info(f"Aggregation health server running on {health_addr}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Aggregation server shutting down…")
    finally:
        if health_server is not None:
            health_server.stop(0)
        server.stop(0)


if __name__ == "__main__":
    serve()
