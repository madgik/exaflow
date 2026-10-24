import asyncio
import threading

import pytest

from exaflow.controller.services.exareme3 import strategies
from exaflow.controller.services.exareme3.strategies import (
    Exareme3WithAggregationServerStrategy,
)
from exaflow.protos.aggregation_server import aggregation_server_pb2 as agg_pb2
from tests.standalone_tests.controller.analysis_support import ResultModel
from tests.standalone_tests.controller.analysis_support import build_strategy
from tests.standalone_tests.controller.analysis_support import patch_algorithm


def test_cancel_during_analysis_cleans_up_only_after_run_finishes(monkeypatch):
    """Client disconnect cancels the request task. Cleanup must wait for run()."""
    order = []
    run_started = threading.Event()
    release_run = threading.Event()

    class FakeAggregationClient:
        def __init__(self, request_id, aggregator_dns=None):
            order.append("init")

        def configure(self, num_workers):
            order.append("configure")
            return agg_pb2.Status.OK

        def cleanup(self):
            order.append("cleanup")
            return agg_pb2.Status.OK

    class BlockingAlgorithm:
        def __init__(self, **kwargs):
            pass

        def run(self):
            order.append("run_start")
            run_started.set()
            release_run.wait(timeout=5)
            order.append("run_end")
            return ResultModel(thread_name="analysis")

    monkeypatch.setattr(
        strategies, "ControllerAggregationClient", FakeAggregationClient
    )
    patch_algorithm(monkeypatch, strategies, BlockingAlgorithm)
    strategy = build_strategy(Exareme3WithAggregationServerStrategy)

    async def scenario():
        task = asyncio.create_task(strategy.execute())
        await asyncio.to_thread(run_started.wait, 5)
        task.cancel()

        async def release():
            await asyncio.sleep(0.1)
            release_run.set()

        asyncio.create_task(release())
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(scenario())
    finally:
        release_run.set()

    assert order == [
        "init",
        "configure",
        "run_start",
        "run_end",
        "cleanup",
    ], order


class _RecordingAggregationClient:
    configure_status = agg_pb2.Status.OK

    def __init__(self, request_id, aggregator_dns=None):
        self.calls = []
        _RecordingAggregationClient.last = self

    def configure(self, num_workers):
        self.calls.append("configure")
        return self.configure_status

    def cleanup(self):
        self.calls.append("cleanup")
        return agg_pb2.Status.OK


def test_refused_configure_does_not_clean_up_the_owning_request(monkeypatch):
    """ALREADY_CONFIGURED means another request with this id owns the context."""

    class AlreadyConfiguredClient(_RecordingAggregationClient):
        configure_status = agg_pb2.Status.ALREADY_CONFIGURED

    monkeypatch.setattr(
        strategies, "ControllerAggregationClient", AlreadyConfiguredClient
    )
    strategy = build_strategy(Exareme3WithAggregationServerStrategy)

    with pytest.raises(RuntimeError, match="refused to configure"):
        asyncio.run(strategy.execute())

    assert AlreadyConfiguredClient.last.calls == ["configure"]


def test_cancel_during_configure_still_cleans_up(monkeypatch):
    configure_started = threading.Event()
    release_configure = threading.Event()

    class BlockingConfigureClient(_RecordingAggregationClient):
        def configure(self, num_workers):
            configure_started.set()
            release_configure.wait(timeout=5)
            return super().configure(num_workers)

    monkeypatch.setattr(
        strategies, "ControllerAggregationClient", BlockingConfigureClient
    )
    strategy = build_strategy(Exareme3WithAggregationServerStrategy)

    async def scenario():
        task = asyncio.create_task(strategy.execute())
        await asyncio.to_thread(configure_started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.1)
        release_configure.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(scenario())
    finally:
        release_configure.set()

    assert BlockingConfigureClient.last.calls == ["configure", "cleanup"]
