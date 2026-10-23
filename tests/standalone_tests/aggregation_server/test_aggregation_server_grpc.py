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
    controller = ControllerAggregationClient(request_id, aggregator_dns=target)
    controller.configure(num_workers)
    return controller


def test_replayed_step_keeps_invalid_argument_status(target):
    endpoint, _ = target
    _configure(endpoint, "replayed-step", 1)
    worker = Exareme3UDFAggregationClient("replayed-step", aggregator_dns=endpoint)
    worker._aggregate_batch_request([(AggregationType.SUM, [1.0])], step=1)

    with pytest.raises(grpc.RpcError) as exc:
        worker._aggregate_batch_request([(AggregationType.SUM, [1.0])], step=1)

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "Step 1 already completed" in exc.value.details()


def test_aggregation_type_mismatch_reaches_all_workers_with_status(target):
    endpoint, _ = target
    _configure(endpoint, "type-mismatch", 2)
    worker_a = Exareme3UDFAggregationClient("type-mismatch", aggregator_dns=endpoint)
    worker_b = Exareme3UDFAggregationClient("type-mismatch", aggregator_dns=endpoint)

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
        if ctx is not None and ctx.batch_payloads:
            return
        time.sleep(0.01)
    raise AssertionError("worker_a payload never arrived before cleanup")


def test_cleanup_aborts_workers_still_waiting(target):
    endpoint, servicer = target
    controller = _configure(endpoint, "cleanup-waiters", 3)
    # worker_a waits for the step 1 result, worker_b waits to start step 2.
    worker_a = Exareme3UDFAggregationClient("cleanup-waiters", aggregator_dns=endpoint)
    worker_b = Exareme3UDFAggregationClient("cleanup-waiters", aggregator_dns=endpoint)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            worker_a._aggregate_batch_request, [(AggregationType.SUM, [1.0])], step=1
        )
        future_b = pool.submit(
            worker_b._aggregate_batch_request, [(AggregationType.SUM, [1.0])], step=2
        )
        _wait_for_payloads(servicer, "cleanup-waiters")
        controller.cleanup()

        # The server waits up to 5s for inputs; Cleanup must release both at once.
        for future in (future_a, future_b):
            with pytest.raises(grpc.RpcError) as exc:
                future.result(timeout=2)
            assert exc.value.code() == grpc.StatusCode.ABORTED
            assert "cancelled" in exc.value.details()


def test_aggregate_after_cleanup_is_aborted_not_invalid_argument(target):
    endpoint, _ = target
    controller = _configure(endpoint, "late-after-cleanup", 1)
    controller.cleanup()
    worker = Exareme3UDFAggregationClient("late-after-cleanup", aggregator_dns=endpoint)

    with pytest.raises(grpc.RpcError) as exc:
        worker.sum([1.0])

    assert exc.value.code() == grpc.StatusCode.ABORTED
    assert "cleaned up" in exc.value.details()


def test_aggregate_without_configure_keeps_invalid_argument(target):
    endpoint, _ = target
    worker = Exareme3UDFAggregationClient("never-configured", aggregator_dns=endpoint)

    with pytest.raises(grpc.RpcError) as exc:
        worker.sum([1.0])

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "not configured" in exc.value.details()
    assert "cleaned up" not in exc.value.details()
