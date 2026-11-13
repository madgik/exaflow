import asyncio
import logging
import types
from concurrent.futures import Future

import grpc
import pytest

from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.controller.services.controller_shutdown import SHUTDOWN_CLEANUP_TIMEOUT
from exaflow.controller.services.errors import ControllerShuttingDownError
from exaflow.controller.services.exareme3 import strategies
from exaflow.protos.aggregation_server import aggregation_server_pb2 as agg_pb2


class _RecordingAggregationClient:
    def __init__(
        self, request_id, context_id, aggregator_dns=None, max_message_mb=None
    ):
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

    def factory(request_id, context_id, aggregator_dns=None, max_message_mb=None):
        client = _RecordingAggregationClient(
            request_id,
            context_id,
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
    strategy._context_id = "ctx"
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
    strategy._context_id = "ctx"
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
        def __init__(
            self, request_id, context_id, aggregator_dns=None, max_message_mb=None
        ):
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
        def Configure(self, request, timeout=None, metadata=None):
            return agg_pb2.ConfigureResponse(status=agg_pb2.Status.OK)

        def Cleanup(self, request, timeout=None, metadata=None):
            raise _DeadlineRpcError(
                grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded"
            )

    client = ControllerAggregationClient(
        "deadline-cleanup", context_id="deadline-cleanup"
    )
    client._stub = DeadlineStub()

    with caplog.at_level(logging.WARNING):
        status = client.cleanup(timeout=0.1)

    assert status == agg_pb2.Status.ERROR
    assert "request_id=deadline-cleanup failed" in caplog.text


def test_client_sends_timeout_as_grpc_deadline():
    seen = {}

    def configure_future(request, timeout=None, metadata=None):
        seen["configure"] = timeout
        future = Future()
        future.set_result(agg_pb2.ConfigureResponse(status=agg_pb2.Status.OK))
        return future

    class RecordingStub:
        Configure = types.SimpleNamespace(future=configure_future)

        def Cleanup(self, request, timeout=None, metadata=None):
            seen["cleanup"] = timeout
            return agg_pb2.CleanupResponse(status=agg_pb2.Status.OK)

    client = ControllerAggregationClient(
        "deadline-pass-through", context_id="deadline-pass-through"
    )
    client._stub = RecordingStub()

    assert client.configure(2, timeout=7) == agg_pb2.Status.OK
    client.cleanup(timeout=7)

    assert seen == {"configure": 7, "cleanup": 7}


class _ConfigureFailsClient:
    """Configure raises ``error``; cleanup records the deadline it was given."""

    error = None
    calls = []

    def __init__(
        self, request_id, context_id, aggregator_dns=None, max_message_mb=None
    ):
        pass

    def configure(self, num_workers, timeout=None):
        self.calls.append("configure")
        raise self.error

    def cleanup(self, timeout=None):
        self.calls.append(("cleanup", timeout))
        return agg_pb2.Status.OK

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.calls.append("close")


@pytest.mark.parametrize(
    "error",
    [grpc.RpcError("deadline exceeded"), ControllerShuttingDownError()],
)
def test_failed_configure_still_cleans_up_with_the_short_deadline(monkeypatch, error):
    """A Configure that timed out or was cancelled may have created the context
    on the server. Cleanup runs, but a slow server does not hold the thread for
    a second full task timeout."""
    client_cls = type(
        "FailingConfigure", (_ConfigureFailsClient,), {"error": error, "calls": []}
    )
    monkeypatch.setattr(strategies, "ControllerAggregationClient", client_cls)
    strategy = _make_strategy(task_timeout=300)

    with pytest.raises(type(error)):
        asyncio.run(strategy.execute())

    assert client_cls.calls == [
        "configure",
        ("cleanup", SHUTDOWN_CLEANUP_TIMEOUT),
        "close",
    ]
