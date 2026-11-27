import asyncio
import logging
import threading
import types

import pytest
from pydantic import BaseModel

from exaflow.algorithms.federated.exceptions import FedErrorReason
from exaflow.algorithms.federated.exceptions import FedObservationError
from exaflow.algorithms.federated.exceptions import FedStateError
from exaflow.algorithms.federated.exceptions import FedTypeError
from exaflow.algorithms.federated.exceptions import FedValueError
from exaflow.controller.services.api.analysis_request_dtos import AnalysisAlgorithmDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisInputDataDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.exareme3 import strategies
from exaflow.controller.services.exareme3.strategies import Exareme3Strategy
from exaflow.worker_communication import AnalysisExecutionError
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError


class _Result(BaseModel):
    thread_name: str


class _RecordingAlgorithm:
    def __init__(self, **kwargs):
        pass

    def run(self):
        return _Result(thread_name=threading.current_thread().name)


@pytest.fixture
def strategy(monkeypatch):
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
    strategy._worker_tasks_handlers = []
    strategy._controller = types.SimpleNamespace(
        worker_landscape_aggregator=types.SimpleNamespace(
            get_metadata=lambda data_model, variable_names: {}
        )
    )
    return strategy


def test_execute_runs_algorithm_off_the_event_loop_thread(strategy):
    async def _execute():
        loop_thread = threading.current_thread().name
        result = await strategy.execute()
        return loop_thread, result

    loop_thread, result = asyncio.run(_execute())

    assert _Result.model_validate_json(result).thread_name != loop_thread


@pytest.mark.parametrize("phase", ["preprocessing", "construction", "execution"])
@pytest.mark.parametrize(
    "source_type, reason, target_type",
    [
        (FedValueError, FedErrorReason.TIMES_POSITIVE_REQUIRED, BadUserInput),
        (FedObservationError, FedErrorReason.INSUFFICIENT_FOLD_SAMPLES, BadUserInput),
        (FedStateError, FedErrorReason.MODEL_NOT_FITTED, RuntimeError),
        (FedTypeError, FedErrorReason.NUMPY_ARRAY_REQUIRED, TypeError),
        (
            FedValueError,
            FedErrorReason.LOGISTIC_CONVERGENCE_FAILED,
            AnalysisExecutionError,
        ),
    ],
)
def test_controller_translates_federated_errors(
    strategy, monkeypatch, phase, source_type, reason, target_type
):
    error = source_type(reason, n_samples=7, n_splits=8)

    class FailingAlgorithm:
        def __init__(self, **kwargs):
            if phase == "construction":
                raise error

        def run(self):
            raise error

    def preprocess(**kwargs):
        raise error

    monkeypatch.setattr(
        strategies, "exareme3_algorithm_classes", {"sample_algo": FailingAlgorithm}
    )
    if phase == "preprocessing":
        monkeypatch.setattr(strategy, "_run_preprocessing_steps", preprocess)

    with pytest.raises(target_type) as exc_info:
        asyncio.run(strategy.execute())

    expected_messages = {
        FedErrorReason.LOGISTIC_CONVERGENCE_FAILED: (
            "Logistic regression could not converge to a stable solution."
        ),
        FedErrorReason.INSUFFICIENT_FOLD_SAMPLES: (
            "There are too few usable observations for the requested number of folds. "
            "Reduce the number of folds or adjust the filters."
        ),
    }
    expected = expected_messages.get(reason, str(error))
    assert type(exc_info.value) is target_type
    assert str(exc_info.value) == expected
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        BadUserInput("Explicit UI message."),
        AnalysisExecutionError("Analysis could not complete."),
        InsufficientDataError("few rows"),
        ValueError("Unexpected failure."),
    ],
)
def test_controller_preserves_other_errors(strategy, monkeypatch, error):
    class FailingAlgorithm(_RecordingAlgorithm):
        def run(self):
            raise error

    monkeypatch.setattr(
        strategies, "exareme3_algorithm_classes", {"sample_algo": FailingAlgorithm}
    )

    with pytest.raises(type(error)) as exc_info:
        asyncio.run(strategy.execute())

    assert exc_info.value is error
