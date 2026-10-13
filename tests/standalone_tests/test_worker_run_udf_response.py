import logging
import math

import pytest

from exaflow.controller.worker_client.tasks_handler import WorkerTasksHandler
from exaflow.protos.worker import worker_pb2
from exaflow.udf_serialization import UDF_RESULT_FORMAT_JSON_BYTES_V1
from exaflow.udf_serialization import decode_udf_payload
from exaflow.udf_serialization import encode_udf_payload
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
    calls = []

    def fake_run_udf(**kwargs):
        calls.append(kwargs)
        return {"f_stat": float("inf")}

    monkeypatch.setattr(grpc_server.udf_service, "run_udf", fake_run_udf)

    service = grpc_server.WorkerService()
    response = service.RunUdf(
        worker_pb2.RunUdfRequest(
            request_id="req",
            udf_registry_key="ols",
            kw_args=encode_udf_payload({"n_splits": 3, "big": 2**60 + 1}),
            system_args=encode_udf_payload({"system": "args"}),
        ),
        context=None,
    )

    assert response.result_format == UDF_RESULT_FORMAT_JSON_BYTES_V1
    decoded = decode_udf_payload(response.result)
    assert decoded["f_stat"] == float("inf")
    kw_args = calls[0]["kw_args"]
    assert kw_args == {"n_splits": 3, "big": 2**60 + 1}
    assert type(kw_args["n_splits"]) is int
    assert calls[0]["system_args"] == {"validated": {"system": "args"}}


def test_worker_tasks_handler_decodes_json_bytes_run_udf_response(monkeypatch):
    client = _FakeWorkerClient(
        worker_pb2.RunUdfResponse(
            result=encode_udf_payload({"items": [1, {"value": 2.5}]}),
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
    assert decode_udf_payload(client.request.kw_args) == {"alpha": 1}
    assert decode_udf_payload(client.request.system_args) == {"metadata": {}}


def test_worker_tasks_handler_decodes_linear_regression_infinite_f_stat(monkeypatch):
    client = _FakeWorkerClient(
        worker_pb2.RunUdfResponse(
            result=encode_udf_payload({"f_stat": float("inf")}),
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
