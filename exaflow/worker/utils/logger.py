import inspect
import logging
import time
from contextvars import ContextVar
from functools import wraps

from exaflow.logging_utils import request_logger
from exaflow.worker import config as worker_config
from exaflow.worker_communication import RequestIDNotFound

LOGGING_ID_TASK_PARAM = "request_id"

_request_logger: ContextVar[logging.LoggerAdapter | None] = ContextVar(
    "_request_logger", default=None
)


def init_logger(request_id):
    return request_logger(
        "exaflow.worker.requests",
        request_id,
        f"%(asctime)s - %(levelname)s - %(module)s.%(funcName)s(%(lineno)d) - [{worker_config.federation}] - [exaflow-{worker_config.role.lower()}] - [{worker_config.identifier}] - [%(request_id)s] - %(message)s",
        worker_config.log_level,
    )


def get_logger():
    logger = _request_logger.get()
    if not logger:
        raise RequestIDNotFound()
    return logger


def initialise_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arglist = inspect.getfullargspec(func)
        if kwargs.get(LOGGING_ID_TASK_PARAM):
            request_id = kwargs.get(LOGGING_ID_TASK_PARAM)
        elif LOGGING_ID_TASK_PARAM in arglist.args:
            # finds the index of request_id arg in list of args from inspect
            # and finds values in args list
            request_id_index = arglist.args.index(LOGGING_ID_TASK_PARAM)
            request_id = args[request_id_index]
        else:
            raise RequestIDNotFound()

        logger = init_logger(request_id)
        token = _request_logger.set(logger)
        try:
            return func(*args, **kwargs)
        finally:
            _request_logger.reset(token)

    return wrapper


def log_method_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        starting_timestamp = time.time()
        logger.info(f"*********** {func.__name__} method started ***********")
        output = func(*args, **kwargs)
        finish_timestamp = time.time()
        tsm_diff = finish_timestamp - starting_timestamp
        logger.info(
            f"*********** {func.__name__} method succeeded in {tsm_diff} ***********"
        )
        return output

    return wrapper
