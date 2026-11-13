"""``run_udf`` returns one call that owns the handler and the gRPC future.

The flow engine starts every worker before it waits. Waiting is
``call.result()``. The handler that opened the RPC and the future stay on
the call, so the engine does not keep a separate finish method.
"""

import concurrent.futures
import logging

from exaflow.controller.services.exareme3.tasks_handler import Exareme3TasksHandler
from exaflow.controller.worker_client.tasks_handler import WorkerTasksHandler
from exaflow.protos.worker import worker_pb2
from exaflow.udf_result_serialization import UDF_RESULT_FORMAT_JSON_BYTES_V1
from exaflow.udf_result_serialization import encode_udf_result
from exaflow.worker_communication import RunUdfSystemArgs


class _Client:
    def __init__(self, response):
        self.response = response
        self.future = concurrent.futures.Future()
        self.future.set_result(response)

    def start_call(self, rpc_name, request, timeout):
        assert rpc_name == "RunUdf"
        return self.future

    def finish_call(self, rpc_name, future):
        assert rpc_name == "RunUdf"
        assert future is self.future
        return future.result()


def test_run_udf_returns_a_call_holding_the_handler_and_the_future(monkeypatch):
    response = worker_pb2.RunUdfResponse(
        result=encode_udf_result({"value": 3}),
        result_format=UDF_RESULT_FORMAT_JSON_BYTES_V1,
    )
    client = _Client(response)
    handler = WorkerTasksHandler("127.0.0.1:1", logging.getLogger("run-udf-call"))
    monkeypatch.setattr(handler, "_client", lambda: client)

    udf_call = handler.run_udf(
        request_id="req",
        udf_registry_key="udf",
        kw_args={},
        system_args={},
        timeout=10,
    )

    assert udf_call.handler is handler
    assert udf_call.future is client.future
    assert udf_call.result() == {"value": 3}
    assert not hasattr(handler, "start_run_udf")
    assert not hasattr(handler, "finish_run_udf")


def test_cancelling_the_call_cancels_its_future(monkeypatch):
    future = concurrent.futures.Future()
    client = _Client(None)
    client.future = future
    handler = WorkerTasksHandler("127.0.0.1:1", logging.getLogger("run-udf-call"))
    monkeypatch.setattr(handler, "_client", lambda: client)

    udf_call = handler.run_udf("req", "udf", {}, {}, 10)

    assert udf_call.cancel() is True
    assert future.cancelled()


def test_exareme3_run_udf_returns_the_worker_call(monkeypatch):
    handler = Exareme3TasksHandler(
        request_id="req",
        worker_id="worker-0",
        worker_queue_addr="127.0.0.1:1",
        tasks_timeout=10,
    )
    sentinel = object()

    def run_udf(**_kwargs):
        return sentinel

    monkeypatch.setattr(handler._worker_tasks_handler, "run_udf", run_udf)
    system_args = RunUdfSystemArgs(
        inputdata={"data_model": "dm", "datasets": ["ds"], "variables": ["x"]},
        metadata={},
        add_dataset_variable=False,
    )

    assert (
        handler.run_udf(
            udf_registry_key="udf", kw_args={"a": 1}, system_args=system_args
        )
        is sentinel
    )
    assert not hasattr(handler, "start_run_udf")
    assert not hasattr(handler, "finish_run_udf")
