import asyncio
import logging
import threading
import time

import pytest

from exaflow.controller import DeploymentType
from exaflow.controller.quart import endpoints
from exaflow.controller.services import exareme3
from exaflow.controller.services.exareme3 import get_exareme3_controller
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.utils import AttrDict


def _make_aggregator(update_interval=60):
    return WorkerLandscapeAggregator(
        logger=logging.getLogger("test_wla_lifecycle"),
        update_interval=update_interval,
        tasks_timeout=1,
        deployment_type=DeploymentType.LOCAL,
        localworkers=AttrDict({}),
    )


class _Counter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self._value += 1

    @property
    def value(self):
        with self._lock:
            return self._value


def test_stop_returns_promptly_while_loop_waits(monkeypatch):
    aggregator = _make_aggregator(update_interval=60)
    # update returns immediately; the loop then waits ~60s for the next refresh.
    monkeypatch.setattr(WorkerLandscapeAggregator, "update", lambda self: None)
    aggregator.start()
    try:
        # Let the loop enter the long wait between refreshes.
        time.sleep(0.2)
        start = time.monotonic()
        aggregator.stop()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, elapsed
        assert not aggregator._update_loop_thread.is_alive()
    finally:
        aggregator.stop()


def test_stop_before_start_is_noop():
    aggregator = _make_aggregator()
    # No thread has been started yet; stop must not raise or block.
    start = time.monotonic()
    aggregator.stop()
    assert time.monotonic() - start < 1.0
    assert aggregator._update_loop_thread is None


def test_stop_during_start_is_honoured(monkeypatch):
    """A stop between clearing the event and starting the thread must stick."""
    aggregator = _make_aggregator(update_interval=60)
    calls = []
    monkeypatch.setattr(aggregator, "update", lambda: calls.append("update"))
    real_clear = threading.Event.clear

    def clear_and_stop(event):
        real_clear(event)
        if event is aggregator._stop_event:
            aggregator.stop()

    monkeypatch.setattr(threading.Event, "clear", clear_and_stop)
    aggregator.start()
    try:
        time.sleep(0.2)
        assert calls == []
        assert not aggregator._update_loop_thread.is_alive()
    finally:
        aggregator.stop()


def test_start_after_stop_resumes_refreshing(monkeypatch):
    counter = _Counter()
    aggregator = _make_aggregator(update_interval=0.05)
    monkeypatch.setattr(WorkerLandscapeAggregator, "update", counter)

    aggregator.start()
    try:
        time.sleep(0.2)  # allow a few refreshes
        assert counter.value >= 1
    finally:
        aggregator.stop()
    stopped_at = counter.value
    time.sleep(0.1)
    assert counter.value == stopped_at  # nothing refreshes while stopped

    aggregator.start()
    try:
        time.sleep(0.2)
        assert counter.value > stopped_at  # refreshing resumed
    finally:
        aggregator.stop()


def test_update_exception_is_logged_and_loop_continues(monkeypatch):
    calls = _Counter()
    aggregator = _make_aggregator(update_interval=0.05)

    def flaky(self):
        calls()
        if calls.value % 2 == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(WorkerLandscapeAggregator, "update", flaky)
    aggregator.start()
    try:
        time.sleep(0.3)
        assert calls.value >= 2  # the loop kept running past the exception
    finally:
        aggregator.stop()


def test_refresh_in_progress_finishes_before_stop_returns(monkeypatch):
    aggregator = _make_aggregator(update_interval=60)
    started = threading.Event()
    release = threading.Event()

    def slow_update(self):
        started.set()
        release.wait(2)

    monkeypatch.setattr(WorkerLandscapeAggregator, "update", slow_update)
    aggregator.start()
    assert started.wait(2)

    stop_done = threading.Event()

    def do_stop():
        aggregator.stop()
        stop_done.set()

    stop_thread = threading.Thread(target=do_stop)
    stop_thread.start()
    try:
        # stop() must not complete while a refresh is still running.
        assert not stop_done.wait(0.2)
        release.set()  # let the in-progress refresh finish
        assert stop_done.wait(2)
    finally:
        release.set()
        aggregator.stop()
        stop_thread.join()


def test_app_shutdown_stops_wla_before_analysis_pool(monkeypatch):
    calls = []
    monkeypatch.setattr(
        endpoints, "stop_background_services", lambda: calls.append("wla")
    )
    monkeypatch.setattr(
        endpoints,
        "shutdown_analysis_execution",
        lambda *_: calls.append("analysis_pool"),
    )

    asyncio.run(endpoints.shutdown())

    assert calls == ["wla", "analysis_pool"]


def test_app_shutdown_stops_the_registered_aggregator(monkeypatch):
    """Shutdown must stop the aggregator that startup registered.

    The order test above replaces ``stop_background_services``. This one
    calls the real helper, which has to resolve the aggregator getter.
    """
    order = []

    class Aggregator:
        def stop(self):
            order.append("wla")

    monkeypatch.setattr(
        "exaflow.controller.services._worker_landscape_aggregator",
        Aggregator(),
    )
    monkeypatch.setattr(
        endpoints,
        "shutdown_analysis_execution",
        lambda *_args: order.append("analysis_pool"),
    )

    asyncio.run(endpoints.shutdown())

    assert order == ["wla", "analysis_pool"]


def test_get_exareme3_controller_raises_when_uninitialized(monkeypatch):
    # Ensure no controller was set for the duration of this test.
    monkeypatch.setattr(exareme3, "_exareme3_controller", None)
    with pytest.raises(
        ValueError, match="Exareme3Controller has not been initialized."
    ):
        get_exareme3_controller()
