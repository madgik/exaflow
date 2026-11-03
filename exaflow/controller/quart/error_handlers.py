import enum
import logging

from quart import Blueprint

from exaflow.controller.logger import get_background_service_logger
from exaflow.controller.services.analysis_execution import AnalysisCapacityExceeded
from exaflow.controller.services.api.analysis_request_validator import BadRequest
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

error_handlers = Blueprint("error_handlers", __name__)

INSUFFICIENT_DATA_ERROR_MESSAGE = (
    "The algorithm could not run with the input "
    "provided because there are insufficient data."
)

_INTERNAL_SERVER_ERROR_TYPES = (WorkerUnresponsiveError, WorkerTaskTimeoutError)


class HTTPStatusCode(enum.IntEnum):
    OK = 200
    BAD_REQUEST = 400
    WORKER_MESSAGE_TOO_LARGE = 413
    BAD_USER_INPUT = 460
    INSUFFICIENT_DATA_ERROR = 461
    SMPC_USAGE_ERROR = 462
    SERVICE_UNAVAILABLE = 503
    WORKER_UNRESPONSIVE_ALGORITHM_EXECUTION_ERROR = 512
    WORKER_TASK_TIMEOUT_ALGORITHM_EXECUTION_ERROR = 513


ERROR_RESPONSES = {
    AnalysisCapacityExceeded: (HTTPStatusCode.SERVICE_UNAVAILABLE, logging.INFO),
    GlobalWorkerUnavailableError: (
        HTTPStatusCode.SERVICE_UNAVAILABLE,
        logging.ERROR,
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


def http_error_body(error: Exception) -> str:
    if isinstance(error, InsufficientDataError):
        return INSUFFICIENT_DATA_ERROR_MESSAGE
    return error.message


def _log_prefix(error_type: type[Exception]) -> str:
    if error_type in _INTERNAL_SERVER_ERROR_TYPES:
        return "Internal Server Error"
    return "Request Error"


def _register_error_response(
    error_type: type[Exception], status: HTTPStatusCode, level: int
) -> None:
    prefix = _log_prefix(error_type)

    @error_handlers.app_errorhandler(error_type)
    def _handle(error: Exception, status=status, level=level, prefix=prefix):
        get_background_service_logger().log(
            level,
            f"{prefix}. Type: '{type(error).__name__}' Message: '{error}'",
        )
        return http_error_body(error), status


for _error_type, (_status, _level) in ERROR_RESPONSES.items():
    _register_error_response(_error_type, _status, _level)
