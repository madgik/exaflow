from __future__ import annotations

import socket
import time
from concurrent import futures

import grpc
import pytest
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from exaflow.protos.worker import worker_pb2
from exaflow.protos.worker import worker_pb2_grpc
from exaflow.worker import config as worker_config
from exaflow.worker.grpc_server import WorkerService
from exaflow.worker.utils import duck_db_csv_loader


def _allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def worker_test_config(monkeypatch, tmp_path_factory) -> dict[str, str | int]:
    """Provide isolated config values so the worker server can start in tests."""
    data_path = tmp_path_factory.mktemp("worker-data")
    duckdb_path = tmp_path_factory.mktemp("duckdb") / "db.duckdb"
    port = _allocate_port()

    monkeypatch.setattr(worker_config, "identifier", "test-worker")
    monkeypatch.setattr(worker_config, "federation", "pytest-federation", raising=False)
    monkeypatch.setattr(worker_config, "data_path", str(data_path))
    worker_config.duckdb.path = str(duckdb_path)
    worker_config.grpc.ip = "127.0.0.1"
    worker_config.grpc.port = port
    worker_config.worker_tasks.tasks_timeout = 5

    return {"listen_addr": f"{worker_config.grpc.ip}:{worker_config.grpc.port}"}


def _start_worker_server(
    monkeypatch,
    listen_addr: str,
    *,
    generic_handlers: tuple[grpc.GenericRpcHandler, ...] | None = None,
) -> grpc.Server:
    # Avoid touching the filesystem-heavy data loader in tests.
    monkeypatch.setattr(
        duck_db_csv_loader,
        "load_all_csvs_from_data_folder",
        lambda request_id: "mocked",
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    worker_service = WorkerService(health_servicer=health_servicer)
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(worker_service, server)

    if generic_handlers:
        server.add_generic_rpc_handlers(generic_handlers)

    port = server.add_insecure_port(listen_addr)
    if port == 0:
        raise RuntimeError(f"Failed to bind test gRPC server to {listen_addr}")

    server.start()
    return server


def _wait_for_healthcheck(
    stub: worker_pb2_grpc.WorkerServiceStub, attempts: int = 10, delay: float = 0.25
):
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            stub.Healthcheck(
                worker_pb2.HealthcheckRequest(request_id="ready", check_db=False),
                timeout=2,
            )
            return
        except (
            grpc.RpcError
        ) as exc:  # pragma: no cover - exercised only on transient startup failures
            last_error = exc
            time.sleep(delay)
    raise AssertionError(f"WorkerService did not become healthy: {last_error}")


def test_worker_service_serves_basic_requests(worker_test_config, monkeypatch):
    listen_addr = worker_test_config["listen_addr"]
    server = _start_worker_server(monkeypatch, listen_addr)
    channel = grpc.insecure_channel(listen_addr)
    stub = worker_pb2_grpc.WorkerServiceStub(channel)

    try:
        _wait_for_healthcheck(stub)
        info_resp = stub.GetWorkerInfo(
            worker_pb2.GetWorkerInfoRequest(request_id="info"), timeout=3
        )
        assert info_resp.worker.id == worker_config.identifier
        assert info_resp.worker.port == worker_config.grpc.port
    finally:
        server.stop(0).wait()
        channel.close()


def test_grpc_channel_recovers_after_server_restart(worker_test_config, monkeypatch):
    listen_addr = worker_test_config["listen_addr"]
    channel = grpc.insecure_channel(listen_addr)
    stub = worker_pb2_grpc.WorkerServiceStub(channel)
    servers: list[grpc.Server] = []

    try:
        server = _start_worker_server(monkeypatch, listen_addr)
        servers.append(server)
        _wait_for_healthcheck(stub)
        stub.Healthcheck(
            worker_pb2.HealthcheckRequest(request_id="before-restart", check_db=False),
            timeout=3,
        )

        server.stop(0).wait()
        time.sleep(0.2)
        with pytest.raises(grpc.RpcError):
            stub.Healthcheck(
                worker_pb2.HealthcheckRequest(
                    request_id="during-downtime", check_db=False
                ),
                timeout=1,
            )

        server = _start_worker_server(monkeypatch, listen_addr)
        servers.append(server)
        _wait_for_healthcheck(stub, attempts=20)

        response = stub.Healthcheck(
            worker_pb2.HealthcheckRequest(request_id="after-restart", check_db=False),
            timeout=3,
        )
        assert response.ok is True
    finally:
        for srv in servers:
            srv.stop(0).wait()
        channel.close()


def test_standard_health_service_recovers(worker_test_config, monkeypatch):
    listen_addr = worker_test_config["listen_addr"]
    server = _start_worker_server(monkeypatch, listen_addr)
    channel = grpc.insecure_channel(listen_addr)
    worker_stub = worker_pb2_grpc.WorkerServiceStub(channel)
    health_stub = health_pb2_grpc.HealthStub(channel)

    try:
        _wait_for_healthcheck(worker_stub)
        health_stub.Check(health_pb2.HealthCheckRequest(service="worker"), timeout=3)

        server.stop(0).wait()
        time.sleep(0.1)
        with pytest.raises(grpc.RpcError):
            health_stub.Check(
                health_pb2.HealthCheckRequest(service="worker"), timeout=1
            )

        server = _start_worker_server(monkeypatch, listen_addr)
        _wait_for_healthcheck(worker_stub, attempts=20)
        health_stub.Check(health_pb2.HealthCheckRequest(service="worker"), timeout=3)
    finally:
        server.stop(0).wait()
        channel.close()


def test_deadline_respected_during_downtime(worker_test_config, monkeypatch):
    listen_addr = worker_test_config["listen_addr"]
    channel = grpc.insecure_channel(listen_addr)
    stub = worker_pb2_grpc.WorkerServiceStub(channel)
    server = _start_worker_server(monkeypatch, listen_addr)

    try:
        _wait_for_healthcheck(stub)
        server.stop(0).wait()
        start = time.perf_counter()
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.Healthcheck(
                worker_pb2.HealthcheckRequest(
                    request_id="deadline-check", check_db=False
                ),
                timeout=0.4,
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0
        assert exc_info.value.code() in {
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.UNAVAILABLE,
        }
    finally:
        server.stop(0).wait()
        channel.close()


@pytest.mark.slow
def test_connectivity_state_transitions(worker_test_config, monkeypatch):
    listen_addr = worker_test_config["listen_addr"]
    channel = grpc.insecure_channel(listen_addr)
    states: list[grpc.ChannelConnectivity] = []
    channel.subscribe(states.append, try_to_connect=True)

    stub = worker_pb2_grpc.WorkerServiceStub(channel)
    server = _start_worker_server(monkeypatch, listen_addr)

    try:
        _wait_for_healthcheck(stub)
        deadline = time.time() + 5
        while grpc.ChannelConnectivity.READY not in states and time.time() < deadline:
            time.sleep(0.05)

        server.stop(0).wait()
        deadline = time.time() + 5
        while (
            states
            and states[-1] == grpc.ChannelConnectivity.READY
            and time.time() < deadline
        ):
            time.sleep(0.05)
        assert any(
            state
            in {
                grpc.ChannelConnectivity.TRANSIENT_FAILURE,
                grpc.ChannelConnectivity.CONNECTING,
                grpc.ChannelConnectivity.SHUTDOWN,
            }
            for state in states
        )

        server = _start_worker_server(monkeypatch, listen_addr)
        deadline = time.time() + 5
        while grpc.ChannelConnectivity.READY not in states and time.time() < deadline:
            time.sleep(0.05)
        assert grpc.ChannelConnectivity.READY in states
    finally:
        server.stop(0).wait()
        channel.close()


@pytest.mark.slow
def test_parallel_calls_recover_after_restart(worker_test_config, monkeypatch):
    listen_addr = worker_test_config["listen_addr"]
    channel = grpc.insecure_channel(listen_addr)
    stub = worker_pb2_grpc.WorkerServiceStub(channel)

    def _call_health(request_id: str, timeout: float = 1.0):
        return stub.Healthcheck(
            worker_pb2.HealthcheckRequest(request_id=request_id, check_db=False),
            timeout=timeout,
        )

    server = _start_worker_server(monkeypatch, listen_addr)
    try:
        _wait_for_healthcheck(stub)
        with futures.ThreadPoolExecutor(max_workers=6) as pool:
            ok_results = list(
                pool.map(lambda idx: _call_health(f"warm-{idx}"), range(6))
            )
        assert all(res.ok for res in ok_results)

        server.stop(0).wait()

        def _attempt(idx: int):
            try:
                _call_health(f"down-{idx}", 0.5)
                return True
            except grpc.RpcError:
                return False

        with futures.ThreadPoolExecutor(max_workers=6) as pool:
            failures = list(pool.map(_attempt, range(6)))
        assert any(not success for success in failures)

        server = _start_worker_server(monkeypatch, listen_addr)
        _wait_for_healthcheck(stub, attempts=20)
        with futures.ThreadPoolExecutor(max_workers=6) as pool:
            ok_after = list(
                pool.map(lambda idx: _call_health(f"up-{idx}", 2.0), range(6))
            )
        assert all(res.ok for res in ok_after)
    finally:
        server.stop(0).wait()
        channel.close()


def test_generic_handler_with_metadata_recovers(worker_test_config, monkeypatch):
    listen_addr = worker_test_config["listen_addr"]

    class MetadataRecorder(grpc.GenericRpcHandler):
        def __init__(self):
            self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

        def service(self, handler_call_details):
            if handler_call_details.method != "/meta.Service/Check":
                return None

            def handler(request, context):
                self.calls.append(
                    (handler_call_details.method, tuple(context.invocation_metadata()))
                )
                return worker_pb2.HealthcheckResponse(ok=True)

            return grpc.unary_unary_rpc_method_handler(
                handler,
                request_deserializer=worker_pb2.HealthcheckRequest.FromString,
                response_serializer=worker_pb2.HealthcheckResponse.SerializeToString,
            )

    recorder = MetadataRecorder()
    server = _start_worker_server(
        monkeypatch, listen_addr, generic_handlers=(recorder,)
    )
    channel = grpc.insecure_channel(listen_addr)
    stub = worker_pb2_grpc.WorkerServiceStub(channel)
    generic_call = channel.unary_unary(
        "/meta.Service/Check",
        request_serializer=worker_pb2.HealthcheckRequest.SerializeToString,
        response_deserializer=worker_pb2.HealthcheckResponse.FromString,
    )

    try:
        _wait_for_healthcheck(stub)
        generic_call(
            worker_pb2.HealthcheckRequest(request_id="before", check_db=False),
            metadata=(("x-req-id", "before"),),
            timeout=2,
        )
        server.stop(0).wait()

        with pytest.raises(grpc.RpcError):
            generic_call(
                worker_pb2.HealthcheckRequest(request_id="down", check_db=False),
                metadata=(("x-req-id", "down"),),
                timeout=1,
            )

        server = _start_worker_server(
            monkeypatch, listen_addr, generic_handlers=(recorder,)
        )
        _wait_for_healthcheck(stub, attempts=20)
        generic_call(
            worker_pb2.HealthcheckRequest(request_id="after", check_db=False),
            metadata=(("x-req-id", "after"),),
            timeout=2,
        )

        # Verify metadata captured before and after restart.
        seen_headers = {dict(md).get("x-req-id") for _, md in recorder.calls}
        assert "before" in seen_headers
        assert "after" in seen_headers
    finally:
        server.stop(0).wait()
        channel.close()
