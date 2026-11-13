"""Worker calls to the aggregation server carry a deadline.

Without one, a worker thread that waits on a hung aggregation server never
returns. The worker passes its task timeout to the client, and the client
passes it to every Aggregate and Unregister call.
"""

import numpy as np
import pytest

import exaflow.protos.aggregation_server.aggregation_server_pb2 as pb2
from exaflow.aggregation_clients.exareme3_udf_aggregation_client import (
    Exareme3UDFAggregationClient,
)
from exaflow.aggregation_serialization import ndarray_to_bytes
from exaflow.worker.exareme3.udf import udf_service

UNSET = object()


class _RecordingStub:
    def __init__(self):
        self.aggregate_timeout = UNSET
        self.unregister_timeout = UNSET

    def Aggregate(self, request, timeout=None):
        self.aggregate_timeout = timeout
        return pb2.AggregateResponse(
            tensors=[
                ndarray_to_bytes(np.asarray([1.0, 2.0])) for _ in request.operations
            ]
        )

    def Unregister(self, request, timeout=None):
        self.unregister_timeout = timeout
        return pb2.UnregisterResponse(status=pb2.Status.OK, remaining_workers=1)


def _client(**kwargs):
    client = Exareme3UDFAggregationClient(
        "req-deadline", aggregator_dns="127.0.0.1:1", **kwargs
    )
    client._stub = _RecordingStub()  # noqa: SLF001
    return client


@pytest.mark.parametrize("operation", ["sum", "min", "max"])
def test_aggregate_calls_carry_the_client_deadline(operation):
    client = _client(timeout=7.5)
    try:
        getattr(client, operation)([1.0, 2.0])
    finally:
        client.close()

    assert client._stub.aggregate_timeout == 7.5  # noqa: SLF001


def test_unregister_carries_the_client_deadline():
    client = _client(timeout=7.5)
    try:
        client.unregister()
    finally:
        client.close()

    assert client._stub.unregister_timeout == 7.5  # noqa: SLF001


def test_client_without_a_deadline_sends_none():
    client = _client()
    try:
        client.sum([1.0, 2.0])
        client.unregister()
    finally:
        client.close()

    assert client._stub.aggregate_timeout is None  # noqa: SLF001
    assert client._stub.unregister_timeout is None  # noqa: SLF001


def test_worker_gives_its_task_timeout_to_the_aggregation_client(monkeypatch):
    created = {}

    class _Client:
        def __init__(self, request_id, **kwargs):
            created["request_id"] = request_id
            created.update(kwargs)

    monkeypatch.setattr(udf_service, "AggregationClient", _Client)
    monkeypatch.setattr(udf_service.worker_config.worker_tasks, "tasks_timeout", 42)
    monkeypatch.setattr(
        udf_service.exareme3_registry, "aggregation_server_required", lambda key: True
    )

    udf_service._create_aggregation_client_if_required(
        request_id="req-deadline",
        udf_registry_key="any_udf",
        preprocessing=[],
    )

    assert created["request_id"] == "req-deadline"
    assert created.get("timeout") == 42
