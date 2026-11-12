import enum
import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import PlainTextResponse

from exaflow.controller.http.request_body_limit import RequestBodyTooLargeError
from exaflow.controller.logger import get_background_service_logger
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

INSUFFICIENT_DATA_ERROR_MESSAGE = (
    "The algorithm could not run with the input "
    "provided because there are insufficient data."
)

_INTERNAL_SERVER_ERROR_TYPES = (WorkerUnresponsiveError, WorkerTaskTimeoutError)


class HTTPStatusCode(enum.IntEnum):
    OK = 200
    BAD_REQUEST = 400
    CONTENT_TOO_LARGE = 413
    UNPROCESSABLE_CONTENT = 422
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
    ControllerShuttingDownError: (
        HTTPStatusCode.SERVICE_UNAVAILABLE,
        logging.INFO,
    ),
    BadRequest: (HTTPStatusCode.BAD_REQUEST, logging.INFO),
    RequestValidationError: (HTTPStatusCode.UNPROCESSABLE_CONTENT, logging.INFO),
    FilterError: (HTTPStatusCode.BAD_REQUEST, logging.INFO),
    BadUserInput: (HTTPStatusCode.BAD_USER_INPUT, logging.INFO),
    DataModelUnavailable: (HTTPStatusCode.BAD_USER_INPUT, logging.INFO),
    DatasetUnavailable: (HTTPStatusCode.BAD_USER_INPUT, logging.INFO),
    InsufficientDataError: (
        HTTPStatusCode.INSUFFICIENT_DATA_ERROR,
        logging.INFO,
    ),
    SMPCUsageError: (HTTPStatusCode.SMPC_USAGE_ERROR, logging.INFO),
    RequestBodyTooLargeError: (HTTPStatusCode.CONTENT_TOO_LARGE, logging.INFO),
    WorkerMessageTooLargeError: (HTTPStatusCode.CONTENT_TOO_LARGE, logging.ERROR),
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
    if isinstance(error, RequestValidationError):
        # Raised by FastAPI for a body that is not JSON or does not match the
        # route's model. The body is plain text, like the other errors.
        details = "\n".join(
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            for detail in error.errors()
        )
        return f"Request malformed:\n{details}"
    return error.message


def _log_prefix(error_type: type[Exception]) -> str:
    if error_type in _INTERNAL_SERVER_ERROR_TYPES:
        return "Internal Server Error"
    return "Request Error"


def _bind_error_handler(
    app: FastAPI, error_type: type[Exception], status: HTTPStatusCode, level: int
) -> None:
    prefix = _log_prefix(error_type)

    async def handler(
        _request, error: Exception, status=status, level=level, prefix=prefix
    ):
        # str() of a RequestValidationError is empty.
        message = (
            http_error_body(error)
            if isinstance(error, RequestValidationError)
            else error
        )
        get_background_service_logger().log(
            level,
            f"{prefix}. Type: '{type(error).__name__}' Message: '{message}'",
        )
        return PlainTextResponse(http_error_body(error), status_code=int(status))

    app.add_exception_handler(error_type, handler)


def register_error_handlers(app: FastAPI) -> None:
    for error_type, (status, level) in ERROR_RESPONSES.items():
        _bind_error_handler(app, error_type, status, level)
