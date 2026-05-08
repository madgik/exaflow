import logging
import math

import pytest

from exaflow.controller.worker_client.tasks_handler import WorkerTasksHandler
from exaflow.protos.worker import worker_pb2
from exaflow.udf_result_serialization import UDF_RESULT_FORMAT_JSON_BYTES_V1
from exaflow.udf_result_serialization import decode_udf_result
from exaflow.udf_result_serialization import encode_udf_result
from exaflow.worker import grpc_server


class _FakeWorkerClient:
    def __init__(self, response):
        self.response = response
        self.request = None

    def call(self, rpc_name, request, timeout):
        self.request = request
        assert rpc_name == "RunUdf"
        assert timeout == 10
        return self.response


class _DummyRunUdfSystemArgs:
    @classmethod
    def model_validate(cls, data):
        return {"validated": data}


def test_worker_service_returns_json_bytes_run_udf_response(monkeypatch):
    monkeypatch.setattr(
        grpc_server.duck_db_csv_loader,
        "load_all_csvs_from_data_folder",
        lambda request_id: "mocked",
    )
    monkeypatch.setattr(grpc_server, "RunUdfSystemArgs", _DummyRunUdfSystemArgs)
    struct_payloads = [{"alpha": 1}, {"system": "args"}]
    monkeypatch.setattr(
        grpc_server,
        "_struct_to_dict",
        lambda struct_message: struct_payloads.pop(0),
    )
    monkeypatch.setattr(
        grpc_server.udf_service,
        "run_udf",
        lambda **kwargs: {"f_stat": float("inf")},
    )

    service = grpc_server.WorkerService()
    response = service.RunUdf(
        worker_pb2.RunUdfRequest(request_id="req", udf_registry_key="ols"),
        context=None,
    )

    assert response.result_format == UDF_RESULT_FORMAT_JSON_BYTES_V1
    decoded = decode_udf_result(response.result)
    assert decoded["f_stat"] == float("inf")


def test_worker_tasks_handler_decodes_json_bytes_run_udf_response(monkeypatch):
    client = _FakeWorkerClient(
        worker_pb2.RunUdfResponse(
            result=encode_udf_result({"items": [1, {"value": 2.5}]}),
            result_format=UDF_RESULT_FORMAT_JSON_BYTES_V1,
        )
    )
    handler = WorkerTasksHandler("127.0.0.1:1", logging.getLogger(__name__))
    monkeypatch.setattr(handler, "_client", lambda: client)

    result = handler.run_udf(
        request_id="req",
        udf_registry_key="udf",
        kw_args={"alpha": 1},
        system_args={"metadata": {}},
        timeout=10,
    )

    assert result == {"items": [1, {"value": 2.5}]}
    assert client.request.request_id == "req"
    assert client.request.udf_registry_key == "udf"


def test_worker_tasks_handler_decodes_linear_regression_infinite_f_stat(monkeypatch):
    client = _FakeWorkerClient(
        worker_pb2.RunUdfResponse(
            result=encode_udf_result({"f_stat": float("inf")}),
            result_format=UDF_RESULT_FORMAT_JSON_BYTES_V1,
        )
    )
    handler = WorkerTasksHandler("127.0.0.1:1", logging.getLogger(__name__))
    monkeypatch.setattr(handler, "_client", lambda: client)

    result = handler.run_udf(
        request_id="req",
        udf_registry_key="linear_regression",
        kw_args={},
        system_args={},
        timeout=10,
    )

    assert math.isinf(result["f_stat"])
    assert result["f_stat"] > 0


def test_worker_tasks_handler_rejects_unknown_run_udf_result_format(monkeypatch):
    client = _FakeWorkerClient(
        worker_pb2.RunUdfResponse(result=b"{}", result_format="unknown")
    )
    handler = WorkerTasksHandler("127.0.0.1:1", logging.getLogger(__name__))
    monkeypatch.setattr(handler, "_client", lambda: client)

    with pytest.raises(ValueError, match="Unsupported RunUdfResponse result format"):
        handler.run_udf(
            request_id="req",
            udf_registry_key="udf",
            kw_args={},
            system_args={},
            timeout=10,
        )
