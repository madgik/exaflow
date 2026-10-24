"""Helpers for controller event-loop and liveness tests."""

import asyncio
import logging
import sys
import threading
import time
import types

from pydantic import BaseModel

from exaflow.controller.quart.app import app
from exaflow.controller.services.api.analysis_request_dtos import AnalysisAlgorithmDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisInputDataDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO

ANALYSIS_BODY = {
    "inputdata": {
        "data_model": "dementia:0.1",
        "datasets": ["ppmi0"],
        "variables": ["x"],
    },
    "algorithm": {"name": "sample_algo", "y": ["x"]},
}


class ResultModel(BaseModel):
    thread_name: str


def set_max_concurrent_experiments(limit: int) -> None:
    """Point the controller at a limit and drop any executor cached so far."""
    from exaflow.controller import config as ctrl_config

    ctrl_config.max_concurrent_experiments = limit
    module = sys.modules.get("exaflow.controller.services.analysis_execution")
    reset = getattr(module, "reset_analysis_execution", None) if module else None
    if reset is not None:
        reset()


def analysis_request() -> AnalysisRequestDTO:
    return AnalysisRequestDTO(
        inputdata=AnalysisInputDataDTO(
            data_model="dementia:0.1",
            datasets=["ppmi0"],
            variables=["x"],
        ),
        algorithm=AnalysisAlgorithmDTO(name="sample_algo", y=["x"]),
    )


def build_strategy(strategy_cls, algorithm_name="sample_algo"):
    strategy = object.__new__(strategy_cls)
    strategy._analysis_request_dto = analysis_request()
    strategy._algorithm_name = algorithm_name
    strategy._request_id = "req"
    strategy._context_id = "ctx"
    strategy._logger = logging.getLogger("test")
    strategy._local_worker_tasks_handlers = []
    strategy._global_worker_tasks_handler = None
    strategy._controller = types.SimpleNamespace(
        worker_landscape_aggregator=types.SimpleNamespace(
            get_metadata=lambda data_model, variable_names: {}
        )
    )
    return strategy


def patch_algorithm(monkeypatch, strategies_module, algorithm_cls, name="sample_algo"):
    monkeypatch.setattr(
        strategies_module, "exareme3_algorithm_classes", {name: algorithm_cls}
    )
    monkeypatch.setattr(
        strategies_module, "log_experiment_execution", lambda *args, **kwargs: None
    )


def run_coroutine_in_thread(coroutine_fn, result):
    def runner():
        try:
            asyncio.run(coroutine_fn())
            result["done"] = True
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


def assert_livez_while_blocked(started, release, drive):
    """GET /livez must succeed while drive() is inside its blocking section."""
    result = {}

    async def scenario():
        async with app.test_client() as client:

            async def poll():
                await asyncio.to_thread(started.wait, 5)
                response = await client.get("/livez")
                result["livez"] = response.status_code

            poll_task = asyncio.create_task(poll())
            response = await drive(client)
            result["analysis"] = response.status_code
            await poll_task

    thread = run_coroutine_in_thread(scenario, result)
    assert started.wait(timeout=3), "blocking section was not entered"
    time.sleep(0.5)
    livez_while_blocked = result.get("livez")
    release.set()
    thread.join(timeout=5)
    assert livez_while_blocked == 200, {
        "livez_while_blocked": livez_while_blocked,
        "after_release": dict(result),
    }
    if result.get("error") is not None:
        raise result["error"]
    assert result.get("done"), result
    assert result.get("analysis") == 200, result
    assert not thread.is_alive()


def assert_loop_responsive_during(async_call, started, release):
    """Heartbeats must keep arriving while async_call is in its blocking section."""
    observed = {"beats": 0}

    async def scenario():
        async def heartbeat():
            while True:
                observed["beats"] += 1
                await asyncio.sleep(0.05)

        pulse = asyncio.create_task(heartbeat())
        try:
            await async_call()
        finally:
            pulse.cancel()
            try:
                await pulse
            except asyncio.CancelledError:
                pass

    result = {}
    thread = run_coroutine_in_thread(scenario, result)
    assert started.wait(timeout=3), "blocking section was not entered"
    baseline = observed["beats"]
    time.sleep(0.4)
    during = observed["beats"] - baseline
    release.set()
    thread.join(timeout=5)
    assert during >= 2, f"event loop stalled during the blocking call ({during})"
    if result.get("error") is not None:
        raise result["error"]
    assert result.get("done"), result
    assert not thread.is_alive()
