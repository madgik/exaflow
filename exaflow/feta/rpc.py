import grpc

from exaflow.feta.lifecycle import ContextClosedError
from exaflow.feta.lifecycle import ResourceLimitError
from exaflow.worker_communication import InsufficientDataError


def abort(context, error):
    if isinstance(error, InsufficientDataError):
        code = grpc.StatusCode.FAILED_PRECONDITION
    elif isinstance(error, ContextClosedError):
        code = grpc.StatusCode.NOT_FOUND
    elif isinstance(error, ResourceLimitError):
        code = grpc.StatusCode.RESOURCE_EXHAUSTED
    elif isinstance(error, TimeoutError):
        code = grpc.StatusCode.DEADLINE_EXCEEDED
    elif isinstance(error, ValueError):
        code = grpc.StatusCode.INVALID_ARGUMENT
    else:
        code = grpc.StatusCode.INTERNAL
    context.abort(code, str(error))
