"""The controller process is Uvicorn, in-process, with the shutdown budget."""

import json
from pathlib import Path

from uvicorn.config import Config

from exaflow.controller.http import server_config

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

    ``--workers 1`` keeps the app as PID 1, so SIGTERM reaches the process
    that runs the lifespan shutdown hook.
    """
    assert server_config.workers == 1
    config = Config(
        "exaflow.controller.http.app:app",
        workers=1,
        timeout_graceful_shutdown=120,
    )
    assert config.workers == 1
    assert config.timeout_graceful_shutdown == 120
    assert not (config.reload or config.workers > 1)
