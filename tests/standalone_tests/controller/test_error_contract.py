"""Failed and malformed analyses surface as the documented errors.

A worker timeout maps to WorkerTaskTimeoutError (HTTP 513), a worker that
fails or cannot be reached maps to WorkerUnresponsiveError (HTTP 512), and a
request without datasets is rejected before any worker is called.
"""

import pydantic
import pytest

from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.errors import ControllerShuttingDownError
from exaflow.controller.services.errors import WorkerTaskTimeoutError
from exaflow.controller.services.errors import WorkerUnresponsiveError
from exaflow.controller.services.exareme3.algorithm_flow_engine_interface import (
    Exareme3AlgorithmFlowEngineInterface,
)
from exaflow.controller.services.exareme3.algorithm_flow_engine_interface import (
    translate_worker_error,
)
from exaflow.controller.worker_client.app import WorkerClientConnectionError
from exaflow.controller.worker_client.app import WorkerClientTimeoutException
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError

ADDRESS = "127.0.0.1:1"


def _timeout():
    return WorkerClientTimeoutException("StatusCode.DEADLINE_EXCEEDED", ADDRESS)


def _connection_error():
    return WorkerClientConnectionError(ADDRESS, "worker crashed")


def _udf(data):
    return {}


class _Call:
    def __init__(self, outcome):
        self._outcome = outcome
        self.cancelled = False

    def add_done_callback(self, fn):
        fn(self)

    def cancel(self):
        self.cancelled = True
        return True

    def result(self):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _Handler:
    def __init__(self, worker_id, outcome=None, start_error=None):
        self.worker_id = worker_id
        self._outcome = outcome
        self._start_error = start_error
        self.call = None

    def run_udf(self, udf_registry_key, kw_args, system_args):
        if self._start_error is not None:
            raise self._start_error
        self.call = _Call(self._outcome)
        return self.call


def _run(handlers):
    engine = Exareme3AlgorithmFlowEngineInterface(
        request_id="req-error-contract",
        context_id="ctx",
        tasks_handlers=handlers,
        inputdata=Inputdata(data_model="dm:1", datasets=["ds"], variables=["x"]),
        metadata={},
    )
    return engine.run_udf(func=_udf, add_dataset_variable=False, kw_args={})


def test_worker_timeout_translates_to_task_timeout_error():
    assert isinstance(translate_worker_error(_timeout()), WorkerTaskTimeoutError)


def test_worker_connection_error_translates_to_unresponsive_error():
    assert isinstance(
        translate_worker_error(_connection_error()), WorkerUnresponsiveError
    )


@pytest.mark.parametrize(
    "error",
    [BadUserInput("bad"), InsufficientDataError("few"), ValueError("other")],
    ids=["bad_user_input", "insufficient_data", "other"],
)
def test_other_errors_are_returned_unchanged(error):
    assert translate_worker_error(error) is error


@pytest.mark.parametrize(
    "worker_error, expected",
    [
        (_timeout(), WorkerTaskTimeoutError),
        (_connection_error(), WorkerUnresponsiveError),
    ],
    ids=["timeout", "connection"],
)
def test_worker_failure_while_waiting_raises_the_documented_error(
    worker_error, expected
):
    healthy = _Handler("worker-0", outcome={"value": 1})
    failing = _Handler("worker-1", outcome=worker_error)

    with pytest.raises(expected) as exc_info:
        _run([healthy, failing])

    assert exc_info.value.__cause__ is worker_error
    assert healthy.call.cancelled is True
    assert failing.call.cancelled is True


@pytest.mark.parametrize(
    "worker_error, expected",
    [
        (_timeout(), WorkerTaskTimeoutError),
        (_connection_error(), WorkerUnresponsiveError),
    ],
    ids=["timeout", "connection"],
)
def test_worker_failure_at_dispatch_raises_the_documented_error(worker_error, expected):
    started = _Handler("worker-0", outcome={"value": 1})
    failing = _Handler("worker-1", start_error=worker_error)

    with pytest.raises(expected) as exc_info:
        _run([started, failing])

    assert exc_info.value.__cause__ is worker_error
    assert started.call.cancelled is True


def test_insufficient_data_worker_is_still_skipped():
    healthy = _Handler("worker-0", outcome={"value": 1})
    small = _Handler("worker-1", outcome=InsufficientDataError("few rows"))

    assert _run([healthy, small]) == [{"value": 1}]


@pytest.mark.parametrize(
    "error",
    [BadUserInput("bad input"), ControllerShuttingDownError()],
    ids=["bad_user_input", "shutting_down"],
)
def test_errors_that_already_have_a_status_code_propagate_unchanged(error):
    failing = _Handler("worker-0", outcome=error)

    with pytest.raises(type(error)) as exc_info:
        _run([failing])

    assert exc_info.value is error


def _analysis_request(datasets):
    return {
        "inputdata": {"data_model": "dm:1", "datasets": datasets, "variables": ["x"]},
        "algorithm": {"name": "describe"},
    }


def test_analysis_request_without_datasets_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="datasets"):
        AnalysisRequestDTO.model_validate(_analysis_request([]))


def test_analysis_request_with_a_dataset_is_accepted():
    request = AnalysisRequestDTO.model_validate(_analysis_request(["ds"]))

    assert request.inputdata.datasets == ["ds"]
