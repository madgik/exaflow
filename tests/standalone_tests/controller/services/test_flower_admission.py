"""Admission guard for Flower analyses.

A Flower analysis must be rejected with HTTP 503 while another Flower
analysis holds the controller lock, instead of queueing on the shared
analysis pool and blocking native analyses. A rejected request makes no
worker calls and releases the analysis slot it was handed.
"""

import asyncio
import logging
import threading
import types

import pytest
from pydantic import BaseModel

from exaflow.controller.http.app import app
from exaflow.controller.services.analysis_execution import AnalysisCapacityExceeded
from exaflow.controller.services.api.analysis_request_dtos import AnalysisAlgorithmDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisInputDataDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.flower.strategies import FLOWER_BUSY_MESSAGE
from exaflow.controller.services.flower.strategies import FlowerStrategy
from tests.standalone_tests.controller.analysis_support import ANALYSIS_BODY
from tests.standalone_tests.controller.analysis_support import (
    set_max_concurrent_experiments,
)


class _Result(BaseModel):
    thread_name: str


class _RecordingHandler:
    """Fake flower worker-task handler that records every worker call."""

    def __init__(self, calls, worker_id="worker-0"):
        self.worker_id = worker_id
        self._calls = calls

    def garbage_collect(self):
        self._calls.append("gc")
        return None

    def start_flower_server(self, *args, **kwargs):
        self._calls.append("server")
        return 111

    def stop_flower_server(self, *args, **kwargs):
        self._calls.append("stop-server")
        return None

    def start_flower_client(self, *args, **kwargs):
        self._calls.append("client")
        return 222

    def stop_flower_client(self, *args, **kwargs):
        self._calls.append("stop-client")
        return None


class _FakeFlowerExecutionInfo:
    def __init__(self):
        self.request_id = None

    def set_inputdata(self, inputdata, request_id):
        self.request_id = request_id

    async def reset(self):
        return None

    async def get_result_with_timeout(self):
        return "{}"


def _build_flower_strategy(controller, calls, handlers):
    strategy = object.__new__(FlowerStrategy)
    strategy._analysis_request_dto = AnalysisRequestDTO(
        inputdata=AnalysisInputDataDTO(
            data_model="dementia:0.1", datasets=["ppmi0"], variables=["x"]
        ),
        algorithm=AnalysisAlgorithmDTO(name="sample_algo", y=["x"]),
    )
    strategy._algorithm_name = "sample_algo"
    strategy._request_id = "req"
    strategy._context_id = "ctx"
    strategy._logger = logging.getLogger("test")
    strategy._local_worker_tasks_handlers = handlers
    strategy._global_worker_tasks_handler = handlers[0] if handlers else None
    strategy._controller = controller
    return strategy


def _make_controller():
    """Return (controller, lock). The lock starts free."""
    lock = asyncio.Lock()
    controller = types.SimpleNamespace(
        algorithm_execution_lock=lock,
        flower_execution_info=_FakeFlowerExecutionInfo(),
        worker_landscape_aggregator=types.SimpleNamespace(
            require_global_worker=lambda: types.SimpleNamespace(ip="1.2.3.4")
        ),
    )
    return controller, lock


def test_flower_rejected_while_another_flower_runs(monkeypatch):
    """Lock held -> 503-style rejection, no worker calls, slot untouched."""
    calls = []
    handler = _RecordingHandler(calls)
    controller, lock = _make_controller()
    strategy = _build_flower_strategy(controller, calls, [handler])

    async def scenario():
        # Simulate another Flower analysis holding the controller lock.
        await lock.acquire()
        try:
            with pytest.raises(AnalysisCapacityExceeded) as excinfo:
                await strategy.execute()
        finally:
            lock.release()
        return excinfo.value

    error = asyncio.run(scenario())

    assert error.message == FLOWER_BUSY_MESSAGE
    assert str(error) == FLOWER_BUSY_MESSAGE
    assert calls == [], f"rejected analysis must not touch workers: {calls}"
    # The rejected request never acquired the lock, so the holder keeps it.
    assert not lock.locked()  # released by our finally, not stolen by execute()


def test_flower_runs_when_lock_is_free(monkeypatch):
    """No other Flower analysis -> it proceeds and talks to the workers."""
    from exaflow.controller.services.flower import strategies as flower_strategies

    calls = []
    handler = _RecordingHandler(calls)
    controller, lock = _make_controller()
    strategy = _build_flower_strategy(controller, calls, [handler])

    monkeypatch.setattr(
        flower_strategies, "flower_algorithm_folder_paths", {"sample_algo": "/fake"}
    )
    monkeypatch.setattr(
        flower_strategies, "log_experiment_execution", lambda *a, **k: None
    )

    async def scenario():
        result = await strategy.execute()
        return result, lock.locked()

    result, locked_after = asyncio.run(scenario())

    assert result == "{}"
    assert controller.flower_execution_info.request_id == "req"
    assert "gc" in calls and "server" in calls and "client" in calls
    # The lock is released on completion, freeing the single Flower slot.
    assert not locked_after


def test_rejected_flower_releases_its_analysis_slot(monkeypatch):
    """The endpoint hands out an analysis slot before execute(); a rejected
    Flower request must give it back so the next analysis can run."""
    from exaflow.controller.http import endpoints
    from exaflow.controller.services.exareme3.strategies import Exareme3Strategy
    from tests.standalone_tests.controller.analysis_support import build_strategy

    set_max_concurrent_experiments(1)

    calls = []
    handler = _RecordingHandler(calls)
    controller, lock = _make_controller()
    busy_strategy = _build_flower_strategy(controller, calls, [handler])

    class _QuickAlgorithm:
        def __init__(self, **kwargs):
            pass

        def run(self):
            return _Result(thread_name=threading.current_thread().name)

    import exaflow.controller.services.exareme3.strategies as exa_strategies

    monkeypatch.setattr(
        exa_strategies, "exareme3_algorithm_classes", {"sample_algo": _QuickAlgorithm}
    )
    monkeypatch.setattr(
        exa_strategies, "log_experiment_execution", lambda *a, **k: None
    )
    working = build_strategy(Exareme3Strategy)

    monkeypatch.setattr(endpoints, "get_worker_landscape_aggregator", lambda: object())
    monkeypatch.setattr(endpoints, "validate_analysis_request", lambda **kwargs: None)
    seen = {"count": 0}

    def select(dto):
        seen["count"] += 1
        return busy_strategy if seen["count"] == 1 else working

    monkeypatch.setattr(endpoints, "get_algorithm_execution_strategy", select)

    async def scenario():
        # Hold the flower lock so the first (flower) request is rejected.
        await lock.acquire()
        try:
            async with app.test_client() as client:
                rejected = await client.post("/analysis", json=ANALYSIS_BODY)
                # A working analysis must still be able to grab the single slot.
                ok = await client.post("/analysis", json=ANALYSIS_BODY)
                return (
                    rejected.status_code,
                    (await rejected.get_data()).decode(),
                    ok.status_code,
                )
        finally:
            lock.release()

    rejected_status, rejected_body, ok_status = asyncio.run(scenario())

    assert rejected_status == 503
    assert rejected_body == FLOWER_BUSY_MESSAGE
    assert calls == [], f"rejected analysis must not touch workers: {calls}"
    assert ok_status == 200, "rejected request leaked its analysis slot"
