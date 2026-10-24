import asyncio
import os
import threading
import time

from exaflow.controller.quart import endpoints
from exaflow.controller.quart.app import app
from exaflow.controller.services.exareme3 import strategies
from exaflow.controller.services.exareme3.strategies import Exareme3Strategy
from tests.standalone_tests.controller.analysis_support import ANALYSIS_BODY
from tests.standalone_tests.controller.analysis_support import ResultModel
from tests.standalone_tests.controller.analysis_support import (
    assert_livez_while_blocked,
)
from tests.standalone_tests.controller.analysis_support import build_strategy
from tests.standalone_tests.controller.analysis_support import patch_algorithm
from tests.standalone_tests.controller.analysis_support import (
    set_max_concurrent_experiments,
)


def _patch_entrypoint(monkeypatch, strategy_factory):
    monkeypatch.setattr(
        endpoints,
        "get_worker_landscape_aggregator",
        lambda: object(),
    )
    monkeypatch.setattr(endpoints, "validate_analysis_request", lambda **kwargs: None)
    monkeypatch.setattr(endpoints, "get_algorithm_execution_strategy", strategy_factory)


def test_excess_concurrent_analyses_are_rejected(monkeypatch):
    set_max_concurrent_experiments(2)
    release = threading.Event()
    state = {"inside": 0, "max_inside": 0}
    guard = threading.Lock()
    two_inside = threading.Event()

    class HoldingStrategy:
        async def execute(self):
            def work():
                with guard:
                    state["inside"] += 1
                    state["max_inside"] = max(state["max_inside"], state["inside"])
                    if state["inside"] >= 2:
                        two_inside.set()
                release.wait(timeout=5)
                with guard:
                    state["inside"] -= 1

            await asyncio.to_thread(work)
            return "{}"

    _patch_entrypoint(monkeypatch, lambda dto: HoldingStrategy())

    async def scenario():
        async with app.test_client() as client:
            first = asyncio.create_task(client.post("/analysis", json=ANALYSIS_BODY))
            second = asyncio.create_task(client.post("/analysis", json=ANALYSIS_BODY))
            await asyncio.to_thread(two_inside.wait, 5)
            third = asyncio.create_task(client.post("/analysis", json=ANALYSIS_BODY))
            done, _pending = await asyncio.wait({third}, timeout=0.4)
            rejected_early = third in done
            status = third.result().status_code if rejected_early else None
            livez = await client.get("/livez")
            release.set()
            first_status = (await first).status_code
            second_status = (await second).status_code
            if not rejected_early:
                status = (await third).status_code
            return {
                "rejected_early": rejected_early,
                "status": status,
                "livez": livez.status_code,
                "first": first_status,
                "second": second_status,
                "max_inside": state["max_inside"],
            }

    try:
        outcome = asyncio.run(scenario())
    finally:
        release.set()

    assert outcome["livez"] == 200
    assert outcome["rejected_early"], outcome
    assert outcome["status"] == 503, outcome
    assert outcome["max_inside"] == 2
    assert outcome["first"] == 200
    assert outcome["second"] == 200


def test_analysis_run_uses_a_bounded_named_executor(monkeypatch):
    set_max_concurrent_experiments(1)
    release = threading.Event()
    entered = threading.Event()
    state = {"inside": 0, "max_inside": 0, "names": []}
    guard = threading.Lock()

    class RecordingAlgorithm:
        def __init__(self, **kwargs):
            pass

        def run(self):
            name = threading.current_thread().name
            with guard:
                state["names"].append(name)
                state["inside"] += 1
                state["max_inside"] = max(state["max_inside"], state["inside"])
                entered.set()
            release.wait(timeout=5)
            with guard:
                state["inside"] -= 1
            return ResultModel(thread_name=name)

    patch_algorithm(monkeypatch, strategies, RecordingAlgorithm)
    first = build_strategy(Exareme3Strategy)
    second = build_strategy(Exareme3Strategy)

    async def scenario():
        first_task = asyncio.create_task(first.execute())
        await asyncio.to_thread(entered.wait, 5)
        second_task = asyncio.create_task(second.execute())
        await asyncio.sleep(0.2)
        overlap = state["max_inside"]
        release.set()
        await first_task
        await second_task
        return overlap

    try:
        overlap = asyncio.run(scenario())
    finally:
        release.set()

    assert state["names"]
    assert all(name.startswith("exaflow-analysis") for name in state["names"]), state
    assert overlap == 1, state


def test_livez_while_algorithm_run_is_blocked(monkeypatch):
    release = threading.Event()
    started = threading.Event()

    class BlockingAlgorithm:
        def __init__(self, **kwargs):
            pass

        def run(self):
            started.set()
            release.wait(timeout=5)
            return ResultModel(thread_name=threading.current_thread().name)

    patch_algorithm(monkeypatch, strategies, BlockingAlgorithm)
    strategy = build_strategy(Exareme3Strategy)
    _patch_entrypoint(monkeypatch, lambda dto: strategy)

    try:
        assert_livez_while_blocked(
            started,
            release,
            lambda client: client.post("/analysis", json=ANALYSIS_BODY),
        )
    finally:
        release.set()


def test_healthcheck_is_not_starved_by_running_analyses(monkeypatch):
    pool_size = min(32, (os.cpu_count() or 1) + 4)
    set_max_concurrent_experiments(pool_size)
    release = threading.Event()
    guard = threading.Lock()
    started = {"count": 0}

    class BlockingAlgorithm:
        def __init__(self, **kwargs):
            pass

        def run(self):
            with guard:
                started["count"] += 1
            release.wait(timeout=5)
            return ResultModel(thread_name=threading.current_thread().name)

    def healthcheck():
        return None

    patch_algorithm(monkeypatch, strategies, BlockingAlgorithm)
    monkeypatch.setattr(
        endpoints,
        "get_worker_landscape_aggregator",
        lambda: type("Wla", (), {"healthcheck": staticmethod(healthcheck)})(),
    )

    async def scenario():
        tasks = [
            asyncio.create_task(build_strategy(Exareme3Strategy).execute())
            for _ in range(pool_size)
        ]
        deadline = time.monotonic() + 3
        while started["count"] < pool_size and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert started["count"] == pool_size
        async with app.test_client() as client:
            health_task = asyncio.create_task(client.get("/healthcheck"))
            done, _pending = await asyncio.wait({health_task}, timeout=0.5)
            healthy = health_task in done and health_task.result().status_code == 200
        release.set()
        await asyncio.gather(*tasks)
        return healthy

    try:
        healthy = asyncio.run(scenario())
    finally:
        release.set()

    assert healthy, "healthcheck did not return while analyses held the executor"
