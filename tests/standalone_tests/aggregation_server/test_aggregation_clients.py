from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

from exaflow.aggregation_clients.base_aggregation_client import BaseAggregationClient
from exaflow.aggregation_clients.constants import AggregationType
from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.aggregation_clients.exareme3_udf_aggregation_client import (
    Exareme3UDFAggregationClient,
)
from exaflow.aggregation_server import config as aggregation_config
from exaflow.aggregation_server.server import AggregationServer
from exaflow.protos.aggregation_server import aggregation_server_pb2 as server_pb2


class InlineRpcError(grpc.RpcError):
    def __init__(self, code, details):
        super().__init__()
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class InlineGrpcContext:
    def abort(self, code, details):
        raise InlineRpcError(code, details)


def _run_parallel(*callables):
    with ThreadPoolExecutor(max_workers=len(callables)) as pool:
        futures = [pool.submit(fn) for fn in callables]
        return [future.result(timeout=5) for future in futures]


def _wait_for_batch_mode(
    servicer: AggregationServer, request_id: str, timeout: float = 1.0
):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ctx = servicer.aggregation_contexts.get(request_id)
        # Wait until the aggregation context has initialized batch operations
        if ctx and ctx.batch_ops:
            return
        time.sleep(0.01)
    raise AssertionError(f"Batch mode not initialized for {request_id}")


@pytest.fixture(autouse=True)
def fast_timeout(monkeypatch):
    monkeypatch.setattr(aggregation_config, "max_wait_for_aggregation_inputs", 0.2)
    yield


@pytest.fixture(autouse=True)
def inline_grpc_stub(monkeypatch):
    servicer = AggregationServer()
    channel = SimpleNamespace(close=lambda: None)

    class InlineStub:
        def Configure(self, request):
            return servicer.Configure(request, InlineGrpcContext())

        def Unregister(self, request):
            return servicer.Unregister(request, InlineGrpcContext())

        def Aggregate(self, request):
            return servicer.Aggregate(request, InlineGrpcContext())

        def Cleanup(self, request):
            return servicer.Cleanup(request, InlineGrpcContext())

    monkeypatch.setattr(
        "exaflow.aggregation_clients.base_aggregation_client.grpc.insecure_channel",
        lambda *_args, **_kwargs: channel,
    )
    monkeypatch.setattr(
        "exaflow.aggregation_clients.base_aggregation_client.pb2_grpc.AggregationServerStub",
        lambda _channel: InlineStub(),
    )
    yield servicer


@pytest.fixture
def controller_factory():
    created: list[ControllerAggregationClient] = []

    def factory(request_id: str) -> ControllerAggregationClient:
        controller = ControllerAggregationClient(request_id)
        created.append(controller)
        return controller

    yield factory

    for controller in created:
        controller.close()


@pytest.fixture
def worker_factory():
    created: list[BaseAggregationClient] = []

    def factory(request_id: str) -> BaseAggregationClient:
        client = BaseAggregationClient(request_id)
        created.append(client)
        return client

    yield factory

    for client in created:
        client.close()


def test_configure_returns_configured_status(controller_factory):
    controller = controller_factory("cfg-success")
    assert controller.configure(1) == server_pb2.Status.OK


def test_duplicate_configure_returns_warning(controller_factory):
    controller = controller_factory("cfg-duplicate")
    controller.configure(1)
    status = controller.configure(1)
    assert status == server_pb2.Status.ALREADY_CONFIGURED


def test_cleanup_without_configuration_returns_no_context(controller_factory):
    controller = controller_factory("cleanup-empty")
    status = controller.cleanup()
    assert status == server_pb2.Status.NOT_FOUND


def test_cleanup_after_configuration_without_activity(controller_factory):
    controller = controller_factory("cleanup-configured")
    controller.configure(1)
    status = controller.cleanup()
    assert status == server_pb2.Status.OK


def test_cleanup_twice_reports_missing_context(controller_factory):
    controller = controller_factory("cleanup-twice")
    controller.configure(1)
    assert controller.cleanup() == server_pb2.Status.OK
    assert controller.cleanup() == server_pb2.Status.NOT_FOUND


def test_single_worker_sum_returns_input(controller_factory, worker_factory):
    request_id = "single-worker"
    controller = controller_factory(request_id)
    controller.configure(1)
    worker = worker_factory(request_id)

    result = worker._aggregate_request(AggregationType.SUM, [1.0, 2.5, -3.0])
    np.testing.assert_allclose(result, np.array([1.0, 2.5, -3.0]))
    controller.cleanup()


def test_multi_worker_sum_combines_vectors(controller_factory, worker_factory):
    request_id = "sum-multi"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    vectors = ([1.0, 2.0, 3.0], [3.5, -1.0, 0.5])
    results = _run_parallel(
        lambda: worker_a._aggregate_request(AggregationType.SUM, vectors[0]),
        lambda: worker_b._aggregate_request(AggregationType.SUM, vectors[1]),
    )
    expected = np.add(*vectors)
    for res in results:
        np.testing.assert_allclose(res, expected)
    controller.cleanup()


def test_multi_worker_min_selects_elementwise_minimum(
    controller_factory, worker_factory
):
    request_id = "min-multi"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    vectors = ([1.0, -4.0, 3.0], [0.5, -2.0, 5.0])
    results = _run_parallel(
        lambda: worker_a._aggregate_request(AggregationType.MIN, vectors[0]),
        lambda: worker_b._aggregate_request(AggregationType.MIN, vectors[1]),
    )
    expected = np.minimum(*vectors)
    for res in results:
        np.testing.assert_allclose(res, expected)


def test_multi_worker_max_selects_elementwise_maximum(
    controller_factory, worker_factory
):
    request_id = "max-multi"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    vectors = ([-1.0, 7.0, 3.5], [0.5, 6.0, 5.0])
    results = _run_parallel(
        lambda: worker_a._aggregate_request(AggregationType.MAX, vectors[0]),
        lambda: worker_b._aggregate_request(AggregationType.MAX, vectors[1]),
    )
    expected = np.maximum(*vectors)
    for res in results:
        np.testing.assert_allclose(res, expected)


def test_reusing_request_id_supports_multiple_rounds(
    controller_factory, worker_factory
):
    request_id = "reuse-rounds"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    for offset in (0, 10):
        vec_a = np.array([1.0 + offset, 2.0 + offset])
        vec_b = np.array([3.0 + offset, -1.0 + offset])
        expected = vec_a + vec_b
        results = _run_parallel(
            lambda: worker_a._aggregate_request(AggregationType.SUM, vec_a),
            lambda: worker_b._aggregate_request(AggregationType.SUM, vec_b),
        )
        for res in results:
            np.testing.assert_allclose(res, expected)


def test_parallel_request_ids_do_not_conflict(controller_factory, worker_factory):
    controller_a = controller_factory("parallel-a")
    controller_b = controller_factory("parallel-b")
    controller_a.configure(1)
    controller_b.configure(1)
    worker_a = worker_factory("parallel-a")
    worker_b = worker_factory("parallel-b")

    results = _run_parallel(
        lambda: worker_a._aggregate_request(AggregationType.MAX, [1.0, 5.0]),
        lambda: worker_b._aggregate_request(AggregationType.MIN, [3.0, -2.0]),
    )
    np.testing.assert_allclose(results[0], np.array([1.0, 5.0]))
    np.testing.assert_allclose(results[1], np.array([3.0, -2.0]))


def test_worker_call_without_configure_aborts(worker_factory):
    worker = worker_factory("missing-config")
    with pytest.raises(InlineRpcError) as exc:
        worker._aggregate_request(AggregationType.SUM, [1.0])
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "not configured" in exc.value.details()


def test_aggregation_type_mismatch_aborts_all_workers(
    controller_factory, worker_factory
):
    request_id = "type-mismatch"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            worker_a._aggregate_request, AggregationType.SUM, [1.0, 2.0]
        )
        future_b = pool.submit(
            worker_b._aggregate_request, AggregationType.MIN, [1.0, 2.0]
        )

        with pytest.raises(InlineRpcError) as exc_b:
            future_b.result(timeout=2)
        assert exc_b.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        with pytest.raises(InlineRpcError) as exc_a:
            future_a.result(timeout=2)
        assert exc_a.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_worker_times_out_when_not_enough_contributions(
    controller_factory, worker_factory
):
    request_id = "partial-timeout"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker = worker_factory(request_id)

    with pytest.raises(InlineRpcError) as exc:
        worker._aggregate_request(AggregationType.SUM, [1.0, 2.0])
    assert exc.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED


def test_batch_aggregation_sum_and_max(controller_factory, worker_factory):
    request_id = "batch-sum-max"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    ops_a = [
        (AggregationType.SUM, [1.0, 1.0, 1.0]),
        (AggregationType.MAX, [2.0, 4.0, 1.0]),
    ]
    ops_b = [
        (AggregationType.SUM, [0.5, 2.0, 3.0]),
        (AggregationType.MAX, [3.0, 1.0, 2.0]),
    ]

    responses = _run_parallel(
        lambda: worker_a._aggregate_batch_request(ops_a),
        lambda: worker_b._aggregate_batch_request(ops_b),
    )
    expected_sum = np.array([1.5, 3.0, 4.0])
    expected_max = np.array([3.0, 4.0, 2.0])
    for response in responses:
        assert len(response) == 2
        np.testing.assert_allclose(response[0], expected_sum)
        np.testing.assert_allclose(response[1], expected_max)


def test_batch_aggregation_union(controller_factory, worker_factory):
    request_id = "batch-union"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    ops_a = [(AggregationType.UNION, ["b", "a", "a"])]
    ops_b = [(AggregationType.UNION, ["c", "a"])]

    responses = _run_parallel(
        lambda: worker_a._aggregate_batch_request(ops_a),
        lambda: worker_b._aggregate_batch_request(ops_b),
    )
    expected_union = np.array(["a", "b", "c"], dtype=object)
    for response in responses:
        assert len(response) == 1
        np.testing.assert_array_equal(response[0], expected_union)


def test_batch_single_worker_request(controller_factory, worker_factory):
    request_id = "batch-single"
    controller = controller_factory(request_id)
    controller.configure(1)
    worker = worker_factory(request_id)

    ops = [
        (AggregationType.MIN, [5.0, 1.0]),
        (AggregationType.MAX, [0.5, 2.0]),
    ]
    response = worker._aggregate_batch_request(ops)
    np.testing.assert_allclose(response[0], np.array([5.0, 1.0]))
    np.testing.assert_allclose(response[1], np.array([0.5, 2.0]))


def test_batch_operation_count_mismatch_aborts(
    controller_factory, worker_factory, inline_grpc_stub
):
    request_id = "batch-count-mismatch"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    ops_a = [
        (AggregationType.SUM, [1.0, 1.0]),
        (AggregationType.MIN, [2.0, 2.0]),
    ]
    ops_b = [(AggregationType.SUM, [1.0, 1.0])]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future_a = pool.submit(worker_a._aggregate_batch_request, ops_a)
        _wait_for_batch_mode(inline_grpc_stub, request_id)
        with pytest.raises(InlineRpcError) as exc_b:
            worker_b._aggregate_batch_request(ops_b)
        assert exc_b.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        with pytest.raises(InlineRpcError) as exc_a:
            future_a.result(timeout=2)
        assert exc_a.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_batch_operation_type_mismatch_aborts(
    controller_factory, worker_factory, inline_grpc_stub
):
    request_id = "batch-type-mismatch"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    ops_a = [
        (AggregationType.SUM, [1.0, 1.0]),
        (AggregationType.MAX, [2.0, 2.0]),
    ]
    ops_b = [
        (AggregationType.SUM, [1.0, 1.0]),
        (AggregationType.MIN, [2.0, 2.0]),
    ]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future_a = pool.submit(worker_a._aggregate_batch_request, ops_a)
        _wait_for_batch_mode(inline_grpc_stub, request_id)
        with pytest.raises(InlineRpcError) as exc_b:
            worker_b._aggregate_batch_request(ops_b)
        assert exc_b.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        with pytest.raises(InlineRpcError) as exc_a:
            future_a.result(timeout=2)
        assert exc_a.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_batch_vector_length_mismatch_surfaces_error(
    controller_factory, worker_factory, inline_grpc_stub
):
    request_id = "batch-vector-mismatch"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    ops_a = [
        (AggregationType.SUM, [1.0, 1.0, 1.0]),
    ]
    ops_b = [
        (AggregationType.SUM, [2.0, 2.0]),
    ]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future_a = pool.submit(worker_a._aggregate_batch_request, ops_a)
        _wait_for_batch_mode(inline_grpc_stub, request_id)

        with pytest.raises(InlineRpcError) as exc_b:
            worker_b._aggregate_batch_request(ops_b)
        assert exc_b.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        with pytest.raises(InlineRpcError) as exc_a:
            future_a.result(timeout=2)
        assert exc_a.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_batch_rounds_can_repeat_with_same_request(controller_factory, worker_factory):
    request_id = "batch-repeat"
    controller = controller_factory(request_id)
    controller.configure(2)
    worker_a = worker_factory(request_id)
    worker_b = worker_factory(request_id)

    for offset in (0, 5):
        ops_a = [
            (AggregationType.MAX, [1.0 + offset, 2.0 + offset]),
        ]
        ops_b = [
            (AggregationType.MAX, [3.0 + offset, -1.0 + offset]),
        ]
        responses = _run_parallel(
            lambda: worker_a._aggregate_batch_request(ops_a),
            lambda: worker_b._aggregate_batch_request(ops_b),
        )
        expected = np.maximum(
            np.array(ops_a[0][1], dtype=np.float64),
            np.array(ops_b[0][1], dtype=np.float64),
        )
        for res in responses:
            np.testing.assert_allclose(res[0], expected)


def test_batch_call_without_configuration_aborts(worker_factory):
    worker = worker_factory("batch-missing-config")
    ops = [(AggregationType.SUM, [1.0, 2.0])]

    with pytest.raises(InlineRpcError) as exc:
        worker._aggregate_batch_request(ops)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_cleanup_handles_offline_server(controller_factory, monkeypatch):
    controller = controller_factory("offline-cleanup")
    controller.configure(1)

    class DummyInactiveError(Exception):
        pass

    def raise_inactive(*_args, **_kwargs):
        raise DummyInactiveError("offline")

    controller._stub.Cleanup = raise_inactive  # type: ignore[attr-defined]
    monkeypatch.setattr(
        grpc,
        "_channel",
        SimpleNamespace(_InactiveRpcError=DummyInactiveError),
        raising=False,
    )

    status = controller.cleanup()
    assert status == server_pb2.Status.ERROR


def test_unregister_reduces_expected_workers(controller_factory):
    request_id = "udf-unregister"
    controller = controller_factory(request_id)
    controller.configure(2)
    client = Exareme3UDFAggregationClient(request_id)

    status, remaining = client.unregister()
    assert status == server_pb2.Status.OK
    assert remaining == 1

    result = client.aggregate(AggregationType.SUM, [1.0, 2.5, -3.0])
    np.testing.assert_allclose(result, np.array([1.0, 2.5, -3.0]))

    controller.cleanup()
    client.close()
