"""The controller process is Uvicorn, in-process, with the shutdown budget."""

import asyncio
import json
import signal
from pathlib import Path

from uvicorn.config import Config

from exaflow.controller.services import controller_shutdown

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_controller_image_command_is_uvicorn_exec_form():
    dockerfile = (REPO_ROOT / "exaflow" / "controller" / "Dockerfile").read_text()
    cmd_line = next(line for line in dockerfile.splitlines() if line.startswith("CMD"))
    assert json.loads(cmd_line[len("CMD ") :]) == [
        "python",
        "-m",
        "uvicorn",
        "exaflow.controller.http.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--workers",
        "1",
        "--timeout-graceful-shutdown",
        "120",
        "--log-config",
        "exaflow/controller/http/log_config.json",
    ]


def test_uvicorn_workers_one_serves_in_process():
    """Uvicorn starts a supervisor only when workers > 1.

    ``--workers 1`` keeps the app as PID 1, so SIGTERM reaches the shutdown
    flag. Hypercorn needed workers = 0 for the same reason.
    """
    config = Config(
        "exaflow.controller.http.app:app",
        workers=1,
        timeout_graceful_shutdown=120,
    )
    assert config.workers == 1
    assert config.timeout_graceful_shutdown == 120
    assert not (config.reload or config.workers > 1)


def test_sigterm_chains_a_signal_module_handler():
    """Uvicorn registers SIGTERM with signal.signal, not the event loop."""
    controller_shutdown.reset_controller_shutdown()
    called = []

    def server_handler(signum, _frame):
        called.append(signum)

    previous = signal.signal(signal.SIGTERM, server_handler)
    try:

        async def scenario():
            controller_shutdown.install_shutdown_signal_handlers()
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)

        asyncio.run(scenario())
        assert called == [signal.SIGTERM]
        assert controller_shutdown.controller_is_shutting_down()
    finally:
        signal.signal(signal.SIGTERM, previous)
        controller_shutdown.reset_controller_shutdown()
