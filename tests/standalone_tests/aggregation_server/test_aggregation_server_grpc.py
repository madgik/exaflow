"""Aggregation server behavior over a real gRPC server.

The inline stub in ``test_aggregation_clients.py`` raises an ``RpcError`` from
``context.abort()``. Real gRPC raises a plain ``Exception`` instead, so status
codes and error propagation must be checked against a real server.
"""

import time
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
from typing import Tuple

import grpc
import pytest

import exaflow.protos.aggregation_server.aggregation_server_pb2 as pb2
from exaflow.aggregation_clients.constants import AggregationType
from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.aggregation_clients.exareme3_udf_aggregation_client import (
    Exareme3UDFAggregationClient,
)
from exaflow.aggregation_server import config as aggregation_config
from exaflow.aggregation_server.server import AggregationServer
from exaflow.protos.aggregation_server.aggregation_server_pb2_grpc import (
    add_AggregationServerServicer_to_server,
)


@pytest.fixture
def target(monkeypatch) -> Iterator[Tuple[str, AggregationServer]]:
    monkeypatch.setattr(aggregation_config, "max_wait_for_aggregation_inputs", 5)
    servicer = AggregationServer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    add_AggregationServerServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield f"127.0.0.1:{port}", servicer
    server.stop(0)


def _configure(target: str, request_id: str, num_workers: int):
    controller = ControllerAggregationClient(
        request_id, aggregator_dns=target, context_id=request_id
    )
    controller.configure(num_workers, timeout=300)
    return controller


def test_replayed_step_keeps_invalid_argument_status(target):
    endpoint, _ = target
    _configure(endpoint, "replayed-step", 1)
    worker = Exareme3UDFAggregationClient(
        "replayed-step", aggregator_dns=endpoint, context_id="replayed-step"
    )
    worker._aggregate_batch_request([(AggregationType.SUM, [1.0])], step=1)

    with pytest.raises(grpc.RpcError) as exc:
        worker._aggregate_batch_request([(AggregationType.SUM, [1.0])], step=1)

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "Step 1 already completed" in exc.value.details()


def test_aggregation_type_mismatch_reaches_all_workers_with_status(target):
    endpoint, _ = target
    _configure(endpoint, "type-mismatch", 2)
    worker_a = Exareme3UDFAggregationClient(
        "type-mismatch", aggregator_dns=endpoint, context_id="type-mismatch"
    )
    worker_b = Exareme3UDFAggregationClient(
        "type-mismatch", aggregator_dns=endpoint, context_id="type-mismatch"
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(worker_a.sum, [1.0])
        future_b = pool.submit(worker_b.min, [1.0])

        for future in (future_a, future_b):
            with pytest.raises(grpc.RpcError) as exc:
                future.result(timeout=2)
            assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "aggregation_type mismatch" in exc.value.details()


def _wait_for_payloads(servicer, request_id, timeout=5.0):
    """Block until worker_a's step 1 input is stored, so Cleanup can't race it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ctx = servicer.aggregation_contexts.get(request_id)
        if ctx is not None and ctx.steps:
            return
        time.sleep(0.01)
    raise AssertionError("worker_a payload never arrived before cleanup")


def test_cleanup_aborts_workers_still_waiting(target):
    endpoint, servicer = target
    controller = _configure(endpoint, "cleanup-waiters", 3)
    # worker_a waits for the step 1 result, worker_b waits to start step 2.
    worker_a = Exareme3UDFAggregationClient(
        "cleanup-waiters", aggregator_dns=endpoint, context_id="cleanup-waiters"
    )
    worker_b = Exareme3UDFAggregationClient(
        "cleanup-waiters", aggregator_dns=endpoint, context_id="cleanup-waiters"
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            worker_a._aggregate_batch_request, [(AggregationType.SUM, [1.0])], step=1
        )
        future_b = pool.submit(
            worker_b._aggregate_batch_request, [(AggregationType.SUM, [1.0])], step=2
        )
        _wait_for_payloads(servicer, "cleanup-waiters")
        controller.cleanup(timeout=300)

        # The server waits up to 5s for inputs; Cleanup must release both at once.
        for future in (future_a, future_b):
            with pytest.raises(grpc.RpcError) as exc:
                future.result(timeout=2)
            assert exc.value.code() == grpc.StatusCode.ABORTED
            assert "cancelled" in exc.value.details()


def test_aggregate_after_cleanup_is_aborted_not_invalid_argument(target):
    endpoint, _ = target
    controller = _configure(endpoint, "late-after-cleanup", 1)
    controller.cleanup(timeout=300)
    worker = Exareme3UDFAggregationClient(
        "late-after-cleanup", aggregator_dns=endpoint, context_id="late-after-cleanup"
    )

    with pytest.raises(grpc.RpcError) as exc:
        worker.sum([1.0])

    assert exc.value.code() == grpc.StatusCode.ABORTED
    assert "cleaned up" in exc.value.details()


def test_aggregate_without_configure_keeps_invalid_argument(target):
    endpoint, _ = target
    worker = Exareme3UDFAggregationClient(
        "never-configured", aggregator_dns=endpoint, context_id="never-configured"
    )

    with pytest.raises(grpc.RpcError) as exc:
        worker.sum([1.0])

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "not configured" in exc.value.details()
    assert "cleaned up" not in exc.value.details()


def test_unknown_aggregation_type_is_rejected_with_its_name(target):
    endpoint, servicer = target
    _configure(endpoint, "unknown-type", 2)
    worker = Exareme3UDFAggregationClient(
        "unknown-type", aggregator_dns=endpoint, context_id="unknown-type"
    )
    request = pb2.AggregateRequest(
        request_id="unknown-type",
        step=1,
        operations=[pb2.Operation(aggregation_type="AVG", tensor=b"x")],
        context_id="unknown-type",
    )

    # Rejected on the first worker's call, without waiting for the second worker.
    with pytest.raises(grpc.RpcError) as exc:
        worker._stub.Aggregate(request, timeout=2)

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "Unsupported aggregation type 'AVG'" in exc.value.details()


def test_attempts_share_request_logging_id_but_isolate_aggregation_state(
    target, caplog
):
    endpoint, servicer = target
    first = ControllerAggregationClient(
        "logical-request", "attempt-one", aggregator_dns=endpoint
    )
    second = ControllerAggregationClient(
        "logical-request", "attempt-two", aggregator_dns=endpoint
    )
    worker = Exareme3UDFAggregationClient(
        "logical-request", "attempt-two", aggregator_dns=endpoint
    )
    try:
        with caplog.at_level("INFO"):
            assert first.configure(1, timeout=5) == pb2.Status.OK
            assert second.configure(1, timeout=5) == pb2.Status.OK
        assert set(servicer.aggregation_contexts) == {"attempt-one", "attempt-two"}
        assert "logical-request" in caplog.text and "attempt-two" in caplog.text
        assert first.cleanup(timeout=5) == pb2.Status.OK
        assert worker.sum([7.0]).tolist() == [7.0]
        late = Exareme3UDFAggregationClient(
            "logical-request", "attempt-one", aggregator_dns=endpoint
        )
        try:
            with pytest.raises(grpc.RpcError) as exc:
                late.sum([1.0])
            assert exc.value.code() == grpc.StatusCode.ABORTED
        finally:
            late.close()
        assert second.cleanup(timeout=5) == pb2.Status.OK
    finally:
        worker.close()
        first.close()
        second.close()
