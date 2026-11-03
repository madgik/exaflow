"""The worker image must let SIGTERM reach the gRPC server.

``CMD ["sh", "-c", ...]`` leaves a shell as PID 1, and that shell does not
forward SIGTERM. The command has to be exec form, and bootstrap.sh has to
replace itself with the server.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_worker_image_command_is_exec_form_without_shell():
    dockerfile = (REPO_ROOT / "exaflow" / "worker" / "Dockerfile").read_text()
    cmd_line = next(line for line in dockerfile.splitlines() if line.startswith("CMD"))

    assert json.loads(cmd_line[len("CMD") :]) == ["/home/bootstrap.sh"]


def test_worker_bootstrap_execs_the_server():
    script = (REPO_ROOT / "exaflow" / "worker" / "bootstrap.sh").read_text()
    commands = [
        line.strip()
        for line in script.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert commands == ["exec python -m exaflow.worker.grpc_server"]
