import logging

from exaflow.controller import BACKGROUND_LOGGER_NAME
from exaflow.controller import config as ctrl_config
from exaflow.logging_utils import request_logger


def get_request_logger(request_id):
    """
    Used for logging information produced after an endpoint request.
    """
    return init_logger(request_id)


def init_logger(request_id, log_level=None):
    return request_logger(
        "exaflow.controller.requests",
        request_id,
        f"%(asctime)s - %(levelname)s - %(module)s.%(funcName)s(%(lineno)d) - [{ctrl_config.federation}] - [exaflow-controller] - [{ctrl_config.node_identifier}] - [%(request_id)s] - %(message)s",
        log_level or ctrl_config.log_level,
    )


def get_background_service_logger() -> logging.Logger:
    """
    Used for logging information produced by any background service.
    """
    return logging.getLogger(BACKGROUND_LOGGER_NAME)


# this is only used by some tests.
# a better implementation needed when logger gets refactored
def set_background_service_logger(log_level):
    logger = logging.getLogger(BACKGROUND_LOGGER_NAME)
    logger.setLevel(log_level)
