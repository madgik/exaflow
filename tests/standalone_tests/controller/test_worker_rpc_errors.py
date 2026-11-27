import grpc
import pytest

from exaflow.controller.worker_client.app import WorkerClient
from exaflow.controller.worker_client.app import WorkerClientConnectionError
from exaflow.worker_communication import AnalysisExecutionError


class _RpcFailure(grpc.RpcError):
    def __init__(self, status):
        self.status = status

    def code(self):
        return self.status

    def details(self):
        return "private worker diagnostic"


@pytest.mark.parametrize(
    "status, expected_type",
    [
        (grpc.StatusCode.INTERNAL, RuntimeError),
        (grpc.StatusCode.UNAVAILABLE, WorkerClientConnectionError),
        (grpc.StatusCode.ABORTED, AnalysisExecutionError),
    ],
)
def test_worker_execution_failure_is_distinct_from_connection_failure(
    status, expected_type
):
    client = object.__new__(WorkerClient)
    client._socket_addr = "worker:50051"
    original = _RpcFailure(status)

    with pytest.raises(expected_type) as exc_info:
        client._handle_rpc_error(original)

    assert type(exc_info.value) is expected_type
    assert exc_info.value.__cause__ is original
    if status == grpc.StatusCode.INTERNAL:
        assert str(exc_info.value) == "Analysis execution failed."
    elif status == grpc.StatusCode.ABORTED:
        assert str(exc_info.value) == original.details()
