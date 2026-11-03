import asyncio
import logging
import types

import grpc
import pytest

from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.controller.services.exareme3 import strategies
from exaflow.protos.aggregation_server import aggregation_server_pb2 as agg_pb2


class _RecordingAggregationClient:
    def __init__(self, request_id, aggregator_dns=None, max_message_mb=None):
        self.calls = []

    def configure(self, num_workers, timeout=None):
        self.calls.append("configure")
        return agg_pb2.Status.OK

    def cleanup(self, timeout=None):
        self.calls.append("cleanup")
        return agg_pb2.Status.OK

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.calls.append("close")


def test_aggregation_strategy_cleans_up_and_closes_client_when_algorithm_fails(
    monkeypatch,
):
    created = []

    def factory(request_id, aggregator_dns=None, max_message_mb=None):
        client = _RecordingAggregationClient(
            request_id,
            aggregator_dns=aggregator_dns,
            max_message_mb=max_message_mb,
        )
        created.append(client)
        return client

    async def failing_execute(self):
        raise ValueError("worker failed")

    monkeypatch.setattr(strategies, "ControllerAggregationClient", factory)
    monkeypatch.setattr(strategies.Exareme3Strategy, "execute", failing_execute)
    strategy = object.__new__(strategies.Exareme3WithAggregationServerStrategy)
    strategy._request_id = "req"
    strategy._local_worker_tasks_handlers = [object(), object()]
    strategy._logger = logging.getLogger(__name__)
    strategy._controller = types.SimpleNamespace(task_timeout=300)

    with pytest.raises(ValueError, match="worker failed"):
        asyncio.run(strategy.execute())

    assert created[-1].calls == [
        "configure",
        "cleanup",
        "close",
    ]


class _DeadlineRpcError(grpc.RpcError):
    def __init__(self, code, details):
        super().__init__()
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


def _make_strategy(task_timeout):
    strategy = object.__new__(strategies.Exareme3WithAggregationServerStrategy)
    strategy._request_id = "req"
    strategy._local_worker_tasks_handlers = [object(), object()]
    strategy._logger = logging.getLogger(__name__)
    strategy._controller = types.SimpleNamespace(task_timeout=task_timeout)
    return strategy


def test_configure_and_cleanup_send_the_controller_task_timeout_as_deadline(
    monkeypatch,
):
    """The deadline handed to the aggregation server is the controller's
    task timeout, for both configure and cleanup."""
    seen = {}

    class DeadlineClient:
        def __init__(self, request_id, aggregator_dns=None, max_message_mb=None):
            pass

        def configure(self, num_workers, timeout=None):
            seen["configure"] = timeout
            return agg_pb2.Status.OK

        def cleanup(self, timeout=None):
            seen["cleanup"] = timeout
            return agg_pb2.Status.OK

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

    async def ok_execute(self):
        return "{}"

    monkeypatch.setattr(strategies, "ControllerAggregationClient", DeadlineClient)
    monkeypatch.setattr(strategies.Exareme3Strategy, "execute", ok_execute)
    strategy = _make_strategy(task_timeout=123)

    asyncio.run(strategy.execute())

    assert seen == {"configure": 123, "cleanup": 123}


def test_cleanup_past_deadline_returns_error_and_logs_warning(caplog):
    """A cleanup that outlives its deadline must release the workers by
    returning ERROR, not hang, and the failure must be visible."""

    class DeadlineStub:
        def Configure(self, request, timeout=None):
            return agg_pb2.ConfigureResponse(status=agg_pb2.Status.OK)

        def Cleanup(self, request, timeout=None):
            raise _DeadlineRpcError(
                grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded"
            )

    client = ControllerAggregationClient("deadline-cleanup")
    client._stub = DeadlineStub()

    with caplog.at_level(logging.WARNING):
        status = client.cleanup(timeout=0.1)

    assert status == agg_pb2.Status.ERROR
    assert "request_id=deadline-cleanup failed" in caplog.text


def test_client_sends_timeout_as_grpc_deadline():
    seen = {}

    class RecordingStub:
        def Configure(self, request, timeout=None):
            seen["configure"] = timeout
            return agg_pb2.ConfigureResponse(status=agg_pb2.Status.OK)

        def Cleanup(self, request, timeout=None):
            seen["cleanup"] = timeout
            return agg_pb2.CleanupResponse(status=agg_pb2.Status.OK)

    client = ControllerAggregationClient("deadline-pass-through")
    client._stub = RecordingStub()

    client.configure(2, timeout=7)
    client.cleanup(timeout=7)

    assert seen == {"configure": 7, "cleanup": 7}


def test_configure_past_deadline_releases_slot_without_cleanup(monkeypatch):
    """A configure that outlives its deadline raises as today; `configured`
    stays False, so cleanup is never called and the analysis slot is freed."""
    calls = []

    class DeadlineConfigureClient:
        def __init__(self, request_id, aggregator_dns=None, max_message_mb=None):
            pass

        def configure(self, num_workers, timeout=None):
            calls.append("configure")
            raise grpc.RpcError("deadline exceeded")

        def cleanup(self, timeout=None):
            calls.append("cleanup")
            return agg_pb2.Status.OK

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            calls.append("close")

    monkeypatch.setattr(
        strategies, "ControllerAggregationClient", DeadlineConfigureClient
    )
    strategy = _make_strategy(task_timeout=1)

    with pytest.raises(grpc.RpcError):
        asyncio.run(strategy.execute())

    assert calls == ["configure", "close"]


def test_configure_deadline_after_success_still_cleans_up(monkeypatch):
    """The server can apply Configure and the client can still see a deadline.

    Cleanup must run, or that context stays until the server's stale eviction.
    """
    calls = []

    class DeadlineAfterSuccess:
        def __init__(self, request_id, aggregator_dns=None, max_message_mb=None):
            pass

        def configure(self, num_workers, timeout=None):
            calls.append("configure")
            raise _DeadlineRpcError(
                grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded"
            )

        def cleanup(self, timeout=None):
            calls.append("cleanup")
            return agg_pb2.Status.OK

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            calls.append("close")

    monkeypatch.setattr(strategies, "ControllerAggregationClient", DeadlineAfterSuccess)
    strategy = _make_strategy(task_timeout=1)

    with pytest.raises(_DeadlineRpcError):
        asyncio.run(strategy.execute())

    assert calls == ["configure", "cleanup", "close"]
