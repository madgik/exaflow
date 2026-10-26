import logging

import pytest

from exaflow import AttrDict
from exaflow.controller.worker_client.app import WorkerClient
from exaflow.controller.worker_client.tasks_handler import WorkerTasksHandler
from exaflow.protos.worker import worker_pb2
from exaflow.protos.worker import worker_pb2_grpc
from exaflow.worker import grpc_server

FIVE_MIB = 5 * 1024 * 1024
ONE_MIB = 1024 * 1024
# Above a 1 MiB configured limit, well under the 100 MiB default.
OVER_ONE_MIB = ONE_MIB + (200 * 1024)


class _LargeResultService(worker_pb2_grpc.WorkerServiceServicer):
    def RunUdf(self, request, context):
        assert len(request.udf_registry_key) == FIVE_MIB
        return worker_pb2.RunUdfResponse(result=b"x" * FIVE_MIB)


class _SmallResultService(worker_pb2_grpc.WorkerServiceServicer):
    def RunUdf(self, request, context):
        return worker_pb2.RunUdfResponse(result=b"ok")


class _OverLimitResultService(worker_pb2_grpc.WorkerServiceServicer):
    def RunUdf(self, request, context):
        return worker_pb2.RunUdfResponse(result=b"x" * OVER_ONE_MIB)


def _set_configured_message_mb(monkeypatch, mb: int) -> None:
    """Set the component config values, not the hard-coded byte copies."""
    # The limit must come from config: an env value the configs do not carry
    # must not reach any channel or server.
    monkeypatch.setenv("GRPC_MAX_MESSAGE_MB", str(mb + 1))

    from exaflow.aggregation_server import config as agg_config
    from exaflow.controller import config as ctrl_config
    from exaflow.worker import config as worker_config

    monkeypatch.setattr(worker_config.grpc, "max_message_mb", mb, raising=False)
    if not hasattr(ctrl_config, "grpc"):
        monkeypatch.setattr(ctrl_config, "grpc", AttrDict({}), raising=False)
    monkeypatch.setattr(ctrl_config.grpc, "max_message_mb", mb, raising=False)
    monkeypatch.setattr(agg_config, "max_message_mb", mb, raising=False)


def _start_server(servicer):
    server = grpc_server._create_grpc_server(max_workers=1)
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def _assert_message_too_large(exc: BaseException) -> None:
    assert type(exc).__name__ == "WorkerMessageTooLargeError"
    text = str(exc).lower()
    assert "too large" in text
    assert "reduce variables" in text
    assert "1048576" in str(exc) or "1 mib" in text


def test_run_udf_messages_above_default_grpc_limit_round_trip():
    server = grpc_server._create_grpc_server(max_workers=1)
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(_LargeResultService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    client = WorkerClient(f"127.0.0.1:{port}")
    try:
        response = client.call(
            "RunUdf",
            worker_pb2.RunUdfRequest(udf_registry_key="k" * FIVE_MIB),
            timeout=30,
        )
        assert len(response.result) == FIVE_MIB
    finally:
        client.close()
        server.stop(0)


def test_over_limit_request_raises_worker_message_too_large(monkeypatch):
    _set_configured_message_mb(monkeypatch, 1)
    server, port = _start_server(_SmallResultService())
    client = WorkerClient(f"127.0.0.1:{port}")
    try:
        with pytest.raises(Exception) as exc_info:
            client.call(
                "RunUdf",
                worker_pb2.RunUdfRequest(udf_registry_key="k" * OVER_ONE_MIB),
                timeout=30,
            )
    finally:
        client.close()
        server.stop(0)

    _assert_message_too_large(exc_info.value)


@pytest.mark.parametrize(
    "details, kind",
    [
        ("CLIENT: Sent message larger than max (2097152 vs. 1048576)", "request"),
        ("SERVER: Received message larger than max (2097152 vs. 1048576)", "request"),
        (
            "Stream removed (CLIENT: Received message larger than max "
            "(2097152 vs. 1048576))",
            "result",
        ),
        ("SERVER: Sent message larger than max (2097152 vs. 1048576)", "result"),
    ],
    ids=["controller_sends", "worker_receives", "controller_receives", "worker_sends"],
)
def test_message_too_large_names_the_oversized_side(details, kind):
    from exaflow.controller.worker_client.app import WorkerMessageTooLargeError

    error = WorkerMessageTooLargeError(
        connection_address="127.0.0.1:1",
        error_details=details,
        max_message_bytes=ONE_MIB,
    )
    assert error.message.startswith(f"{kind} too large (2097152 bytes")


def test_over_limit_response_raises_worker_message_too_large(monkeypatch):
    _set_configured_message_mb(monkeypatch, 1)
    server, port = _start_server(_OverLimitResultService())
    client = WorkerClient(f"127.0.0.1:{port}")
    try:
        with pytest.raises(Exception) as exc_info:
            client.call(
                "RunUdf",
                worker_pb2.RunUdfRequest(udf_registry_key="small"),
                timeout=30,
            )
    finally:
        client.close()
        server.stop(0)

    _assert_message_too_large(exc_info.value)


def test_large_kw_args_and_result_round_trip_through_run_udf(monkeypatch):
    blob = "v" * (256 * 1024)
    result_blob = "r" * (256 * 1024)

    monkeypatch.setattr(
        grpc_server.duck_db_csv_loader,
        "load_all_csvs_from_data_folder",
        lambda request_id: None,
    )

    def fake_run_udf(**kwargs):
        assert kwargs["kw_args"]["blob"] == blob
        assert kwargs["system_args"].inputdata.data_model == "dementia"
        return {"blob": kwargs["kw_args"]["blob"], "result": result_blob}

    monkeypatch.setattr(grpc_server.udf_service, "run_udf", fake_run_udf)

    server, port = _start_server(grpc_server.WorkerService())
    address = f"127.0.0.1:{port}"
    handler = WorkerTasksHandler(address, logging.getLogger("grpc-message-size"))
    try:
        result = handler.run_udf(
            request_id="req-large",
            udf_registry_key="size_probe",
            kw_args={"blob": blob},
            system_args={
                "inputdata": {
                    "data_model": "dementia",
                    "datasets": ["ppmi0"],
                    "variables": ["x"],
                },
                "metadata": {"x": {"is_categorical": False}},
                "check_min_rows": False,
                "add_dataset_variable": False,
            },
            timeout=30,
        )
    finally:
        handler._client().close()
        server.stop(0)

    assert result == {"blob": blob, "result": result_blob}


def test_configured_limit_reaches_channel_and_server_options(monkeypatch):
    _set_configured_message_mb(monkeypatch, 2)
    expected = [
        ("grpc.max_send_message_length", 2 * ONE_MIB),
        ("grpc.max_receive_message_length", 2 * ONE_MIB),
    ]
    captured = {"channels": [], "servers": []}

    def capture_channel(*args, **kwargs):
        captured["channels"].append(kwargs.get("options"))
        return _FakeChannel()

    def capture_server(*args, **kwargs):
        captured["servers"].append(kwargs.get("options"))
        return _FakeServer()

    monkeypatch.setattr(
        "exaflow.controller.worker_client.app.grpc.insecure_channel",
        capture_channel,
    )
    monkeypatch.setattr(
        "exaflow.aggregation_clients.base_aggregation_client.grpc.insecure_channel",
        capture_channel,
    )
    monkeypatch.setattr("exaflow.worker.grpc_server.grpc.server", capture_server)
    monkeypatch.setattr("exaflow.aggregation_server.server.grpc.server", capture_server)

    WorkerClient("127.0.0.1:1")
    grpc_server._create_grpc_server(max_workers=1)

    from exaflow.aggregation_server.server import serve
    from exaflow.worker.exareme3.udf import udf_service

    monkeypatch.setattr(
        udf_service.exareme3_registry, "aggregation_server_required", lambda _: True
    )
    udf_service._create_aggregation_client_if_required(
        request_id="req-size", udf_registry_key="udf", preprocessing=None
    )
    serve()

    assert captured["channels"], "controller and aggregation channels were not opened"
    assert captured["servers"], "worker and aggregation servers were not created"
    for options in captured["channels"] + captured["servers"]:
        assert options == expected


class _FakeChannel:
    def unary_unary(self, *args, **kwargs):
        return lambda *call_args, **call_kwargs: None

    def close(self):
        return None


class _FakeServer:
    def add_insecure_port(self, address):
        return 1

    def start(self):
        return None

    def stop(self, grace):
        return None

    def wait_for_termination(self, timeout=None):
        return True

    def add_generic_rpc_handlers(self, handlers):
        return None

    def add_registered_method_handlers(self, service_name, handlers):
        return None


def test_worker_grpc_config_uses_max_message_mb():
    import exaflow.worker as worker_config

    configured = AttrDict({"data_path": "/tmp", "grpc": {"max_message_mb": "6"}})
    worker_config._ensure_grpc_config(configured)
    assert configured.grpc.max_message_mb == 6

    defaulted = AttrDict({"data_path": "/tmp", "grpc": {}})
    worker_config._ensure_grpc_config(defaulted)
    assert defaulted.grpc.max_message_mb == 100


def test_controller_config_defaults_max_message_mb():
    from exaflow.controller import config as ctrl_config

    grpc_cfg = getattr(ctrl_config, "grpc", None)
    assert getattr(grpc_cfg, "max_message_mb", None) == 100


def test_aggregation_config_defaults_max_message_mb(monkeypatch):
    import exaflow.aggregation_server as agg_server

    monkeypatch.delenv("GRPC_MAX_MESSAGE_MB", raising=False)
    monkeypatch.delenv("AGG_SERVER_CONFIG_FILE", raising=False)
    loaded = agg_server._load_config()
    assert loaded.max_message_mb == 100


def test_aggregation_config_reads_max_message_mb_from_env(monkeypatch, tmp_path):
    import exaflow.aggregation_server as agg_server

    monkeypatch.setenv("GRPC_MAX_MESSAGE_MB", "6")
    cfg_path = tmp_path / "aggregation.toml"
    cfg_path.write_text('max_message_mb = "$GRPC_MAX_MESSAGE_MB"\n')
    monkeypatch.setenv("AGG_SERVER_CONFIG_FILE", str(cfg_path))
    loaded = agg_server._load_config()
    assert loaded.max_message_mb == 6
    assert isinstance(loaded.max_message_mb, int)


def test_aggregation_config_treats_blank_max_message_mb_as_default(
    monkeypatch, tmp_path
):
    import exaflow.aggregation_server as agg_server

    # Helm renders a missing grpc_max_message_mb value as an empty string.
    monkeypatch.setenv("GRPC_MAX_MESSAGE_MB", "")
    cfg_path = tmp_path / "aggregation.toml"
    cfg_path.write_text('max_message_mb = "$GRPC_MAX_MESSAGE_MB"\n')
    monkeypatch.setenv("AGG_SERVER_CONFIG_FILE", str(cfg_path))
    assert agg_server._load_config().max_message_mb == 100
