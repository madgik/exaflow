"""Shutdown must bound in-flight RPCs instead of waiting out worker_tasks_timeout."""

import asyncio
import logging
import signal
import threading
from types import SimpleNamespace

import grpc
import pytest

from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.controller import DeploymentType
from exaflow.controller.quart import hypercorn_config as hc
from exaflow.controller.services import controller_shutdown as shutdown
from exaflow.controller.services.analysis_execution import reset_analysis_execution
from exaflow.controller.services.analysis_execution import run_blocking
from exaflow.controller.services.exareme3 import strategies
from exaflow.controller.services.exareme3.strategies import (
    Exareme3WithAggregationServerStrategy,
)
from exaflow.controller.services.startup import stop_background_services
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.controller.worker_client.app import WorkerClient
from exaflow.protos.aggregation_server import aggregation_server_pb2 as agg_pb2
from exaflow.utils import AttrDict
from tests.standalone_tests.controller.analysis_support import ResultModel
from tests.standalone_tests.controller.analysis_support import build_strategy
from tests.standalone_tests.controller.analysis_support import patch_algorithm


class _BlockingCall:
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


def test_shutdown_cancel_aborts_an_inflight_rpc():
    shutdown.reset_controller_shutdown()
    call = _BlockingCall()
    shutdown.register_inflight_rpc(call)
    started = threading.Event()

    def block():
        started.set()
        return call.result()

    async def scenario():
        shutdown.begin_controller_shutdown()
        task = asyncio.create_task(run_blocking(block))
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)

    try:
        asyncio.run(scenario())
    finally:
        call.cancel()
        shutdown.reset_controller_shutdown()
        reset_analysis_execution()

    assert call.cancelled


def test_shutdown_cleanup_uses_the_short_deadline(monkeypatch):
    shutdown.reset_controller_shutdown()

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
        shutdown.reset_controller_shutdown()
        reset_analysis_execution()

    assert timeouts == [hc.shutdown_cleanup_timeout]


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

    assert seen["join_timeout"] == hc.shutdown_cleanup_timeout
    assert seen["join_timeout"] <= hc.shutdown_timeout


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


def test_worker_rpc_is_registered_for_shutdown_cancel():
    shutdown.reset_controller_shutdown()
    client = WorkerClient("127.0.0.1:1")
    call = _BlockingCall()

    def future(request, timeout=None):
        return call

    client._stub = SimpleNamespace(RunUdf=SimpleNamespace(future=future))
    try:
        client.start_call("RunUdf", object(), 1)
        shutdown.cancel_inflight_rpcs()
        assert call.cancelled
    finally:
        client.close()
        shutdown.reset_controller_shutdown()


def test_aggregation_configure_unblocks_when_its_rpc_is_cancelled():
    shutdown.reset_controller_shutdown()
    client = ControllerAggregationClient("req-shutdown")
    call = _BlockingCall()

    def future(request, timeout=None):
        return call

    client._stub = SimpleNamespace(Configure=SimpleNamespace(future=future))
    errors = []

    def run():
        try:
            client.configure(1, timeout=30)
        except grpc.FutureCancelledError:
            errors.append("cancelled")

    thread = threading.Thread(target=run)
    thread.start()
    try:
        for _ in range(50):
            if call._callbacks:
                break
            threading.Event().wait(0.01)
        shutdown.cancel_inflight_rpcs()
        thread.join(1)
        assert call.cancelled
        assert not thread.is_alive()
        assert errors == ["cancelled"]
    finally:
        call.cancel()
        thread.join(1)
        client.close()
        shutdown.reset_controller_shutdown()


def test_sigterm_marks_shutdown_and_keeps_the_server_handler():
    shutdown.reset_controller_shutdown()
    called = []

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, lambda: called.append("hypercorn"))
        try:
            shutdown.install_shutdown_signal_handlers()
            handle = loop._signal_handlers[signal.SIGTERM]
            handle._callback(*handle._args)
        finally:
            loop.remove_signal_handler(signal.SIGTERM)

    try:
        asyncio.run(scenario())
        assert called == ["hypercorn"]
        assert shutdown.controller_is_shutting_down()
    finally:
        shutdown.reset_controller_shutdown()
