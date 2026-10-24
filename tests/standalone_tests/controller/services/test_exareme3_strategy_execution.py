import asyncio
import logging
import threading
import types

from pydantic import BaseModel

from exaflow.controller.services.api.analysis_request_dtos import AnalysisAlgorithmDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisInputDataDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.exareme3 import strategies
from exaflow.controller.services.exareme3.strategies import Exareme3Strategy


class _Result(BaseModel):
    thread_name: str


class _RecordingAlgorithm:
    def __init__(self, **kwargs):
        pass

    def run(self):
        return _Result(thread_name=threading.current_thread().name)


def test_execute_runs_algorithm_off_the_event_loop_thread(monkeypatch):
    monkeypatch.setattr(
        strategies, "exareme3_algorithm_classes", {"sample_algo": _RecordingAlgorithm}
    )
    monkeypatch.setattr(strategies, "log_experiment_execution", lambda *a, **k: None)

    strategy = object.__new__(Exareme3Strategy)
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
    strategy._local_worker_tasks_handlers = []
    strategy._controller = types.SimpleNamespace(
        worker_landscape_aggregator=types.SimpleNamespace(
            get_metadata=lambda data_model, variable_names: {}
        )
    )

    async def _execute():
        loop_thread = threading.current_thread().name
        result = await strategy.execute()
        return loop_thread, result

    loop_thread, result = asyncio.run(_execute())

    assert _Result.model_validate_json(result).thread_name != loop_thread
