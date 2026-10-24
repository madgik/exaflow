import asyncio
import threading

import pytest

from exaflow.controller.quart import endpoints
from exaflow.controller.services.exareme3 import strategies as exareme3_strategies
from exaflow.controller.services.exareme3.strategies import (
    Exareme3WithAggregationServerStrategy,
)
from exaflow.controller.services.flower import strategies as flower_strategies
from exaflow.controller.services.flower.strategies import FlowerStrategy
from exaflow.protos.aggregation_server import aggregation_server_pb2 as agg_pb2
from tests.standalone_tests.controller.analysis_support import ANALYSIS_BODY
from tests.standalone_tests.controller.analysis_support import ResultModel
from tests.standalone_tests.controller.analysis_support import (
    assert_livez_while_blocked,
)
from tests.standalone_tests.controller.analysis_support import (
    assert_loop_responsive_during,
)
from tests.standalone_tests.controller.analysis_support import build_strategy
from tests.standalone_tests.controller.analysis_support import patch_algorithm


def _patch_entrypoint(monkeypatch, strategy):
    monkeypatch.setattr(
        endpoints,
        "get_worker_landscape_aggregator",
        lambda: object(),
    )
    monkeypatch.setattr(endpoints, "validate_analysis_request", lambda **kwargs: None)
    monkeypatch.setattr(
        endpoints, "get_algorithm_execution_strategy", lambda dto: strategy
    )


class _FastAlgorithm:
    def __init__(self, **kwargs):
        pass

    def run(self):
        return ResultModel(thread_name="analysis")


def _aggregation_strategy(monkeypatch, configure, cleanup):
    class FakeAggregationClient:
        def __init__(self, request_id, aggregator_dns=None):
            pass

        def configure(self, num_workers):
            return configure(num_workers)

        def cleanup(self):
            return cleanup()

    monkeypatch.setattr(
        exareme3_strategies, "ControllerAggregationClient", FakeAggregationClient
    )
    patch_algorithm(monkeypatch, exareme3_strategies, _FastAlgorithm)
    return build_strategy(Exareme3WithAggregationServerStrategy)


def test_livez_during_aggregation_configure(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def configure(num_workers):
        started.set()
        release.wait(timeout=5)
        return agg_pb2.Status.OK

    strategy = _aggregation_strategy(
        monkeypatch, configure, lambda: agg_pb2.Status.OK
    )
    _patch_entrypoint(monkeypatch, strategy)
    try:
        assert_livez_while_blocked(
            started,
            release,
            lambda client: client.post("/analysis", json=ANALYSIS_BODY),
        )
    finally:
        release.set()


def test_livez_during_aggregation_cleanup(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def cleanup():
        started.set()
        release.wait(timeout=5)
        return agg_pb2.Status.OK

    strategy = _aggregation_strategy(
        monkeypatch, lambda num_workers: agg_pb2.Status.OK, cleanup
    )
    _patch_entrypoint(monkeypatch, strategy)
    try:
        assert_livez_while_blocked(
            started,
            release,
            lambda client: client.post("/analysis", json=ANALYSIS_BODY),
        )
    finally:
        release.set()


def _flower_strategy(monkeypatch, blocked_call, started, release):
    monkeypatch.setattr(
        flower_strategies,
        "flower_algorithm_folder_paths",
        {"sample_algo": "/tmp/flower-algo"},
    )
    monkeypatch.setattr(
        flower_strategies, "log_experiment_execution", lambda *args, **kwargs: None
    )

    def call(name, result=None):
        def fn(*args, **kwargs):
            if name == blocked_call:
                started.set()
                release.wait(timeout=5)
            return result

        return fn

    local = type("Local", (), {})()
    local.worker_id = "local-1"
    local.garbage_collect = call("garbage_collect")
    local.start_flower_client = call("start_flower_client", 11)
    local.stop_flower_client = call("stop_flower_client")

    global_handler = type("Global", (), {})()
    global_handler.garbage_collect = call("garbage_collect_global")
    global_handler.start_flower_server = call("start_flower_server", 10)
    global_handler.stop_flower_server = call("stop_flower_server")

    if blocked_call == "garbage_collect":
        global_handler.garbage_collect = call("garbage_collect")

    strategy = build_strategy(FlowerStrategy)
    strategy._local_worker_tasks_handlers = [local]
    strategy._global_worker_tasks_handler = global_handler

    async def get_result():
        return "{}"

    async def reset():
        return None

    strategy._controller.algorithm_execution_lock = asyncio.Lock()
    strategy._controller.flower_execution_info = type("Info", (), {})()
    strategy._controller.flower_execution_info.set_inputdata = lambda inputdata: None
    strategy._controller.flower_execution_info.get_result_with_timeout = get_result
    strategy._controller.flower_execution_info.reset = reset
    strategy._controller.worker_landscape_aggregator.get_global_worker = (
        lambda: type("Worker", (), {"ip": "127.0.0.1"})()
    )
    return strategy


@pytest.mark.parametrize(
    "blocked_call",
    [
        "garbage_collect",
        "start_flower_server",
        "start_flower_client",
        "stop_flower_server",
        "stop_flower_client",
    ],
)
def test_flower_worker_calls_do_not_block_the_event_loop(
    monkeypatch, blocked_call
):
    started = threading.Event()
    release = threading.Event()
    strategy = _flower_strategy(monkeypatch, blocked_call, started, release)
    try:
        assert_loop_responsive_during(strategy.execute, started, release)
    finally:
        release.set()


def test_livez_during_flower_garbage_collect(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    strategy = _flower_strategy(monkeypatch, "garbage_collect", started, release)
    _patch_entrypoint(monkeypatch, strategy)
    try:
        assert_livez_while_blocked(
            started,
            release,
            lambda client: client.post("/analysis", json=ANALYSIS_BODY),
        )
    finally:
        release.set()
