"""Controller errors are one table, not a handler per exception."""

import logging

from exaflow.controller.http.error_handlers import ERROR_RESPONSES
from exaflow.controller.http.error_handlers import INSUFFICIENT_DATA_ERROR_MESSAGE
from exaflow.controller.http.error_handlers import HTTPStatusCode
from exaflow.controller.http.error_handlers import http_error_body
from exaflow.controller.services.analysis_execution import AnalysisCapacityExceeded
from exaflow.controller.services.api.analysis_request_validator import BadRequest
from exaflow.controller.services.errors import ControllerShuttingDownError
from exaflow.controller.services.errors import GlobalWorkerUnavailableError
from exaflow.controller.services.errors import WorkerTaskTimeoutError
from exaflow.controller.services.errors import WorkerUnresponsiveError
from exaflow.controller.worker_client.app import WorkerMessageTooLargeError
from exaflow.data_filters import FilterError
from exaflow.smpc_cluster_communication import SMPCUsageError
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import DataModelUnavailable
from exaflow.worker_communication import DatasetUnavailable
from exaflow.worker_communication import InsufficientDataError


def test_error_responses_are_one_table():
    assert not hasattr(HTTPStatusCode, "UNEXPECTED_ERROR")
    assert ERROR_RESPONSES == {
        AnalysisCapacityExceeded: (HTTPStatusCode.SERVICE_UNAVAILABLE, logging.INFO),
        GlobalWorkerUnavailableError: (
            HTTPStatusCode.SERVICE_UNAVAILABLE,
            logging.ERROR,
        ),
        ControllerShuttingDownError: (
            HTTPStatusCode.SERVICE_UNAVAILABLE,
            logging.INFO,
        ),
        BadRequest: (HTTPStatusCode.BAD_REQUEST, logging.INFO),
        FilterError: (HTTPStatusCode.BAD_REQUEST, logging.INFO),
        BadUserInput: (HTTPStatusCode.BAD_USER_INPUT, logging.INFO),
        DataModelUnavailable: (HTTPStatusCode.BAD_USER_INPUT, logging.INFO),
        DatasetUnavailable: (HTTPStatusCode.BAD_USER_INPUT, logging.INFO),
        InsufficientDataError: (
            HTTPStatusCode.INSUFFICIENT_DATA_ERROR,
            logging.INFO,
        ),
        SMPCUsageError: (HTTPStatusCode.SMPC_USAGE_ERROR, logging.INFO),
        WorkerMessageTooLargeError: (
            HTTPStatusCode.WORKER_MESSAGE_TOO_LARGE,
            logging.ERROR,
        ),
        WorkerUnresponsiveError: (
            HTTPStatusCode.WORKER_UNRESPONSIVE_ALGORITHM_EXECUTION_ERROR,
            logging.ERROR,
        ),
        WorkerTaskTimeoutError: (
            HTTPStatusCode.WORKER_TASK_TIMEOUT_ALGORITHM_EXECUTION_ERROR,
            logging.ERROR,
        ),
    }


def test_insufficient_data_keeps_its_fixed_message():
    error = InsufficientDataError("row count")
    assert http_error_body(error) == INSUFFICIENT_DATA_ERROR_MESSAGE
    assert http_error_body(BadRequest("bad request")) == "bad request"
