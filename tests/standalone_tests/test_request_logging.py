import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from exaflow import AttrDict
from exaflow.controller import _ensure_log_levels
from exaflow.controller.logger import get_request_logger
from exaflow.worker.utils import logger as worker_logging
from exaflow.worker_communication import RequestIDNotFound


@pytest.mark.parametrize("value", [None, "", "  "])
def test_empty_controller_log_levels_default_to_info(value):
    cfg = AttrDict({"log_level": value, "framework_log_level": value})
    _ensure_log_levels(cfg)
    assert cfg.log_level == "INFO"
    assert cfg.framework_log_level == "INFO"


@pytest.mark.parametrize("level", ["DEBUG", "WARNING", 0])
def test_explicit_controller_log_levels_are_preserved(level):
    cfg = AttrDict({"log_level": level, "framework_log_level": level})
    _ensure_log_levels(cfg)
    assert cfg.log_level == level
    assert cfg.framework_log_level == level


def test_controller_logging_works_without_log_level_environment_variables():
    env = dict(os.environ)
    for name in ("LOG_LEVEL", "FRAMEWORK_LOG_LEVEL", "EXAFLOW_CONTROLLER_CONFIG_FILE"):
        env.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging
from exaflow.controller import config
from exaflow.controller.http.loggers import loggers
from exaflow.controller.logger import get_request_logger

assert config.log_level == config.framework_log_level == "INFO"
assert loggers["handlers"]["framework"]["level"] == "INFO"
assert get_request_logger("unset-level").logger.level == logging.INFO
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("factory", [get_request_logger, worker_logging.init_logger])
def test_unique_requests_reuse_logger_and_handler(factory, monkeypatch):
    # Other suites can attach pytest capture handlers while importing workers.
    # Exercise our handler initialization independently of that instrumentation.
    base = factory("setup").logger
    monkeypatch.setattr(base, "handlers", [])
    first = factory("first")
    registry_size = len(logging.Logger.manager.loggerDict)
    handlers = list(first.logger.handlers)
    for index in range(1000):
        adapter = factory(f"unique-analysis-{index}")
        assert adapter.logger is first.logger
        assert adapter.extra["request_id"] == f"unique-analysis-{index}"
    assert len(logging.Logger.manager.loggerDict) == registry_size
    assert first.logger.handlers == handlers
    assert len(handlers) == 1


@pytest.mark.parametrize("factory", [get_request_logger, worker_logging.init_logger])
def test_concurrent_requests_keep_their_logging_context(factory, monkeypatch):
    records = []
    barrier = threading.Barrier(2)
    base = factory("setup").logger

    class Capture(logging.Handler):
        def emit(self, record):
            records.append((record.request_id, record.getMessage()))

    monkeypatch.setattr(base, "handlers", [Capture()])

    @worker_logging.initialise_logger
    def emit(request_id):
        adapter = factory(request_id)
        barrier.wait(timeout=5)
        adapter.info("message for %s", request_id)
        assert worker_logging.get_logger().extra["request_id"] == request_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(emit, ["alpha", "beta"]))
    assert sorted(records) == [
        ("alpha", "message for alpha"),
        ("beta", "message for beta"),
    ]


def test_worker_context_is_restored_after_nested_failure():
    @worker_logging.initialise_logger
    def inner(request_id):
        assert worker_logging.get_logger().extra["request_id"] == "inner"
        raise ValueError("failed")

    @worker_logging.initialise_logger
    def outer(request_id):
        with pytest.raises(ValueError):
            inner("inner")
        assert worker_logging.get_logger().extra["request_id"] == "outer"

    outer("outer")
    with pytest.raises(RequestIDNotFound):
        worker_logging.get_logger()


@pytest.mark.parametrize("factory", [get_request_logger, worker_logging.init_logger])
def test_control_characters_are_escaped_in_request_context(factory):
    assert factory("id\n\r\t\u2028").extra["request_id"] == r"id\n\r\t\u2028"
