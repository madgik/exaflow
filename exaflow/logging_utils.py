"""Request context without permanently registering one logger per request."""

import logging
import threading

_configuration_lock = threading.Lock()


def safe_log_value(value: object) -> str:
    return "".join(
        char if char.isprintable() else ascii(char)[1:-1] for char in str(value)
    )


def request_logger(name, request_id, format_string, level):
    logger = logging.getLogger(name)
    with _configuration_lock:
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(handler)
            logger.propagate = False
        logger.setLevel(level)
    return logging.LoggerAdapter(logger, {"request_id": safe_log_value(request_id)})
