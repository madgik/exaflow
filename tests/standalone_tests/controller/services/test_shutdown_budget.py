"""Shutdown must cancel what runs an analysis, and only that, once the drain is over."""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import grpc
import pytest

from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.controller import DeploymentType
from exaflow.controller.quart import endpoints
from exaflow.controller.services import controller_shutdown as shutdown
from exaflow.controller.services.analysis_execution import reset_analysis_execution
from exaflow.controller.services.analysis_execution import run_blocking
from exaflow.controller.services.analysis_execution import shutdown_analysis_execution
from exaflow.controller.services.errors import ControllerShuttingDownError
from exaflow.controller.services.exareme3 import strategies
from exaflow.controller.services.exareme3.strategies import (
    Exareme3WithAggregationServerStrategy,
)
from exaflow.controller.services.startup import stop_background_services
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.controller.worker_client.app import WorkerClient
from exaflow.controller.worker_client.tasks_handler import WorkerTasksHandler
from exaflow.protos.aggregation_server import aggregation_server_pb2 as agg_pb2
from exaflow.utils import AttrDict
from tests.standalone_tests.controller.analysis_support import ResultModel
from tests.standalone_tests.controller.analysis_support import build_strategy
from tests.standalone_tests.controller.analysis_support import patch_algorithm


class _BlockingCall:
    """A gRPC future that blocks in ``result`` until it is cancelled."""

    def __init__(self):
        self._done = threading.Event()
        self.cancelled = False
        self._callbacks = []

    def cancel(self):
        self.cancelled = True
        self._done.set()
        for callback in list(self._callbacks):
            callback(self)
        return True

    def result(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError()
        if self.cancelled:
            raise grpc.FutureCancelledError()
        return "ok"

    def add_done_callback(self, fn):
        self._callbacks.append(fn)
        if self._done.is_set():
            fn(self)


def test_client_disconnect_still_waits_for_the_analysis_thread():
    started = threading.Event()
    release = threading.Event()

    def block():
        started.set()
        release.wait(2)
        return "done"

    async def scenario():
        task = asyncio.create_task(run_blocking(block))
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.1)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        reset_analysis_execution()


def test_client_disconnect_does_not_cancel_other_analyses():
    """SIGTERM starts a drain, not a shutdown: only the lifespan hook cancels.

    A disconnect while other analyses drain must leave their RPCs alone.
    """
    other_analysis = _BlockingCall()
    shutdown.register_inflight_rpc(other_analysis)
    started = threading.Event()
    release = threading.Event()

    def block():
        started.set()
        release.wait(2)

    async def scenario():
        task = asyncio.create_task(run_blocking(block))
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.1)
        assert not other_analysis.cancelled
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        reset_analysis_execution()

    assert not other_analysis.cancelled


def test_begin_shutdown_cancels_the_registered_rpcs():
    first, second = _BlockingCall(), _BlockingCall()
    shutdown.register_inflight_rpc(first)
    shutdown.register_inflight_rpc(second)

    shutdown.begin_controller_shutdown()

    assert first.cancelled and second.cancelled
    assert shutdown.controller_is_shutting_down()


def test_rpc_registered_after_shutdown_began_is_cancelled_at_once():
    """An analysis thread that starts its next batch after the cancel stops too."""
    shutdown.begin_controller_shutdown()
    late = _BlockingCall()

    shutdown.register_inflight_rpc(late)

    assert late.cancelled


def test_a_finished_rpc_is_not_cancelled_by_a_later_shutdown():
    finished = _BlockingCall()
    shutdown.register_inflight_rpc(finished)
    finished._callbacks[0](finished)  # gRPC runs the done callbacks on completion

    shutdown.begin_controller_shutdown()

    assert not finished.cancelled


def test_shutdown_hook_cancels_rpcs_before_stopping_anything(monkeypatch):
    calls = []
    call = _BlockingCall()
    shutdown.register_inflight_rpc(call)
    monkeypatch.setattr(
        endpoints,
        "stop_background_services",
        lambda: calls.append(("wla", call.cancelled)),
    )
    monkeypatch.setattr(
        endpoints,
        "shutdown_analysis_execution",
        lambda *_args: calls.append(("pool", call.cancelled)),
    )

    asyncio.run(endpoints.shutdown())

    assert calls == [("wla", True), ("pool", True)]


def test_shutdown_cleanup_uses_the_short_deadline(monkeypatch):
    timeouts = []

    class FakeAggregationClient:
        def __init__(self, request_id, aggregator_dns=None, max_message_mb=None):
            pass

        def configure(self, num_workers, timeout=None):
            return agg_pb2.Status.OK

        def cleanup(self, timeout=None):
            timeouts.append(timeout)
            return agg_pb2.Status.OK

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

    class FastAlgorithm:
        def __init__(self, **kwargs):
            pass

        def run(self):
            return ResultModel(thread_name="analysis")

    monkeypatch.setattr(
        strategies, "ControllerAggregationClient", FakeAggregationClient
    )
    patch_algorithm(monkeypatch, strategies, FastAlgorithm)
    strategy = build_strategy(Exareme3WithAggregationServerStrategy)

    try:
        shutdown.begin_controller_shutdown()
        asyncio.run(strategy.execute())
    finally:
        reset_analysis_execution()

    assert timeouts == [shutdown.SHUTDOWN_CLEANUP_TIMEOUT]


def test_background_stop_bounds_the_wla_join(monkeypatch):
    seen = {}

    class Aggregator:
        def stop(self, join_timeout=None):
            seen["join_timeout"] = join_timeout

    monkeypatch.setattr(
        "exaflow.controller.services.startup.get_worker_landscape_aggregator",
        lambda: Aggregator(),
    )

    stop_background_services()

    assert seen["join_timeout"] == shutdown.SHUTDOWN_CLEANUP_TIMEOUT
    assert seen["join_timeout"] <= shutdown.SHUTDOWN_TIMEOUT


def test_wla_stop_join_timeout_returns_while_a_refresh_is_blocked(monkeypatch):
    aggregator = WorkerLandscapeAggregator(
        logger=logging.getLogger("test_shutdown_budget"),
        update_interval=60,
        tasks_timeout=300,
        deployment_type=DeploymentType.LOCAL,
        localworkers=AttrDict({}),
    )
    started = threading.Event()
    release = threading.Event()

    def slow_update(self):
        started.set()
        release.wait(5)

    monkeypatch.setattr(WorkerLandscapeAggregator, "update", slow_update)
    aggregator.start()
    assert started.wait(2)

    errors = []
    stop_done = threading.Event()

    def do_stop():
        try:
            aggregator.stop(join_timeout=0.05)
        except TypeError as exc:
            errors.append(exc)
        stop_done.set()

    stop_thread = threading.Thread(target=do_stop)
    stop_thread.start()
    try:
        assert stop_done.wait(1), "stop(join_timeout=...) waited out the refresh"
        assert not errors
    finally:
        release.set()
        aggregator.stop()
        stop_thread.join(2)


def test_analysis_pool_join_is_bounded_and_warns_about_a_stuck_thread(caplog):
    started = threading.Event()
    release = threading.Event()

    def stuck():
        started.set()
        release.wait(5)

    async def scenario():
        task = asyncio.create_task(run_blocking(stuck))
        await asyncio.to_thread(started.wait, 2)
        began = time.monotonic()
        with caplog.at_level(logging.WARNING):
            await asyncio.to_thread(shutdown_analysis_execution, 0.2)
        elapsed = time.monotonic() - began
        release.set()
        await task
        return elapsed

    try:
        elapsed = asyncio.run(scenario())
    finally:
        release.set()
        reset_analysis_execution()

    assert elapsed < 2
    assert "analysis threads still running" in caplog.text


def test_run_udf_rpc_is_registered_for_shutdown_cancel(monkeypatch):
    call = _BlockingCall()
    handler = WorkerTasksHandler("127.0.0.1:1", logging.getLogger(__name__))
    client = SimpleNamespace(start_call=lambda *_args: call)
    monkeypatch.setattr(handler, "_client", lambda: client)

    handler.start_run_udf("req", "udf", {}, {}, 1)
    shutdown.begin_controller_shutdown()

    assert call.cancelled


def test_cleanup_and_metadata_rpcs_are_not_cancelled_by_shutdown():
    """GarbageCollect, health and metadata calls finish on their own deadline."""
    call = _BlockingCall()
    client = WorkerClient("127.0.0.1:1")
    client._stub = SimpleNamespace(
        GarbageCollect=SimpleNamespace(future=lambda request, timeout=None: call)
    )
    try:
        future = client.start_call("GarbageCollect", object(), 1)
        shutdown.begin_controller_shutdown()
        assert not call.cancelled
        assert future is call
    finally:
        client.close()


def test_aggregation_cleanup_rpc_is_not_cancelled_by_shutdown():
    client = ControllerAggregationClient("req-cleanup")
    seen = {}

    def cleanup(request, timeout=None):
        seen["shutting_down"] = shutdown.controller_is_shutting_down()
        return agg_pb2.CleanupResponse(status=agg_pb2.Status.OK)

    client._stub = SimpleNamespace(Cleanup=cleanup)
    try:
        shutdown.begin_controller_shutdown()
        assert client.cleanup(timeout=5) == agg_pb2.Status.OK
        assert seen == {"shutting_down": True}
    finally:
        client.close()


def test_cancelled_worker_rpc_raises_the_shutting_down_error():
    client = WorkerClient("127.0.0.1:1")
    call = _BlockingCall()
    call.cancel()
    try:
        with pytest.raises(ControllerShuttingDownError):
            client.finish_call("RunUdf", call)
    finally:
        client.close()


def test_aggregation_configure_unblocks_when_its_rpc_is_cancelled():
    client = ControllerAggregationClient("req-shutdown")
    call = _BlockingCall()

    def future(request, timeout=None):
        return call

    client._stub = SimpleNamespace(Configure=SimpleNamespace(future=future))
    errors = []

    def run():
        try:
            client.configure(1, timeout=30)
        except ControllerShuttingDownError:
            errors.append("cancelled")

    thread = threading.Thread(target=run)
    thread.start()
    try:
        for _ in range(50):
            if call._callbacks:
                break
            threading.Event().wait(0.01)
        shutdown.begin_controller_shutdown()
        thread.join(1)
        assert call.cancelled
        assert not thread.is_alive()
        assert errors == ["cancelled"]
    finally:
        call.cancel()
        thread.join(1)
        client.close()
