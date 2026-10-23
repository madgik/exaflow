"""Aggregation server behavior over a real gRPC server.

The inline stub in ``test_aggregation_clients.py`` raises an ``RpcError`` from
``context.abort()``. Real gRPC raises a plain ``Exception`` instead, so status
codes and error propagation must be checked against a real server.
"""

import time
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

from exaflow.aggregation_clients.base_aggregation_client import BaseAggregationClient
from exaflow.aggregation_clients.constants import AggregationType
from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.aggregation_server import config as aggregation_config
from exaflow.aggregation_server.server import AggregationServer
from exaflow.protos.aggregation_server.aggregation_server_pb2_grpc import (
    add_AggregationServerServicer_to_server,
)


@pytest.fixture
def target(monkeypatch):
    monkeypatch.setattr(aggregation_config, "max_wait_for_aggregation_inputs", 5)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    add_AggregationServerServicer_to_server(AggregationServer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield f"127.0.0.1:{port}"
    server.stop(0)


def _configure(target: str, request_id: str, num_workers: int):
    controller = ControllerAggregationClient(request_id, aggregator_dns=target)
    controller.configure(num_workers)
    return controller


def test_replayed_step_keeps_invalid_argument_status(target):
    _configure(target, "replayed-step", 1)
    worker = BaseAggregationClient("replayed-step", aggregator_dns=target)
    worker._aggregate_request(AggregationType.SUM, [1.0], step=1)

    with pytest.raises(grpc.RpcError) as exc:
        worker._aggregate_request(AggregationType.SUM, [1.0], step=1)

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "Step 1 already completed" in exc.value.details()


def test_aggregation_type_mismatch_reaches_all_workers_with_status(target):
    _configure(target, "type-mismatch", 2)
    worker_a = BaseAggregationClient("type-mismatch", aggregator_dns=target)
    worker_b = BaseAggregationClient("type-mismatch", aggregator_dns=target)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(worker_a._aggregate_request, AggregationType.SUM, [1.0])
        future_b = pool.submit(worker_b._aggregate_request, AggregationType.MIN, [1.0])

        for future in (future_a, future_b):
            with pytest.raises(grpc.RpcError) as exc:
                future.result(timeout=2)
            assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "aggregation_type mismatch" in exc.value.details()


def test_cleanup_aborts_workers_still_waiting(target):
    controller = _configure(target, "cleanup-waiters", 3)
    # worker_a waits for the step 1 result, worker_b waits to start step 2.
    worker_a = BaseAggregationClient("cleanup-waiters", aggregator_dns=target)
    worker_b = BaseAggregationClient("cleanup-waiters", aggregator_dns=target)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            worker_a._aggregate_request, AggregationType.SUM, [1.0], step=1
        )
        future_b = pool.submit(
            worker_b._aggregate_request, AggregationType.SUM, [1.0], step=2
        )
        time.sleep(0.3)
        controller.cleanup()

        # The server waits up to 5s for inputs; Cleanup must release both at once.
        for future in (future_a, future_b):
            with pytest.raises(grpc.RpcError) as exc:
                future.result(timeout=2)
            assert exc.value.code() == grpc.StatusCode.ABORTED
            assert "cancelled" in exc.value.details()
