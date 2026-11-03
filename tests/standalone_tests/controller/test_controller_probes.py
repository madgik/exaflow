import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from hypercorn.config import Config

REPO_ROOT = Path(__file__).resolve().parents[3]


def _controller_block(rendered: str) -> str:
    start = rendered.index("      - name: controller\n")
    end = rendered.index("### --- SMPC", start)
    return rendered[start:end]


def _probes(block: str) -> dict:
    probes = {}
    current = None
    for line in block.splitlines():
        stripped = line.strip()
        if stripped in {"startupProbe:", "livenessProbe:", "readinessProbe:"}:
            current = stripped[:-1]
            probes[current] = {}
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split(":", 1))
        if key in {"path", "timeoutSeconds", "failureThreshold", "periodSeconds"}:
            probes[current][key] = value
    return probes


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_controller_probes_keep_liveness_on_livez():
    rendered = subprocess.check_output(
        ["helm", "template", str(REPO_ROOT / "kubernetes")],
        text=True,
    )
    block = _controller_block(rendered)
    probes = _probes(block)

    assert probes["startupProbe"]["path"] == "/livez"
    assert probes["startupProbe"]["timeoutSeconds"] == "5"
    assert probes["livenessProbe"]["path"] == "/livez"
    assert probes["livenessProbe"]["timeoutSeconds"] == "5"
    assert probes["livenessProbe"]["failureThreshold"] == "3"
    assert probes["readinessProbe"]["path"] == "/healthcheck"
    assert block.count("path: /healthcheck") == 1

    env_at = block.index("name: MAX_CONCURRENT_EXPERIMENTS")
    assert 'value: "32"' in block[env_at : env_at + 80]


def _controller_deployment_block(rendered: str) -> str:
    start = rendered.index("name: exaflow-controller-deployment")
    end = rendered.index("\n---", start)
    return rendered[start:end]


def test_hypercorn_graceful_timeout_is_120_seconds():
    config = Config.from_object("exaflow.controller.quart.hypercorn_config")

    assert config.graceful_timeout == 120


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_controller_pod_grace_period_covers_hypercorn_shutdown():
    rendered = subprocess.check_output(
        ["helm", "template", str(REPO_ROOT / "kubernetes")],
        text=True,
    )
    block = _controller_deployment_block(rendered)
    grace = int(re.search(r"terminationGracePeriodSeconds: (\d+)", block).group(1))
    timeout_at = block.index("name: WORKER_TASKS_TIMEOUT")
    worker_tasks_timeout = int(
        re.search(r'value: "(\d+)"', block[timeout_at : timeout_at + 80]).group(1)
    )
    config = Config.from_object("exaflow.controller.quart.hypercorn_config")
    margin = 20

    # A worker RPC can start near the end of the drain and still run for the
    # full per-call timeout. The pod must stay up through that call, the
    # drain, the shutdown hooks, and a small margin.
    assert worker_tasks_timeout == 300
    assert config.graceful_timeout == 120
    assert config.shutdown_timeout == 60
    assert grace == (
        worker_tasks_timeout
        + config.graceful_timeout
        + config.shutdown_timeout
        + margin
    )


def test_controller_image_command_is_exec_form_without_shell():
    dockerfile = (REPO_ROOT / "exaflow" / "controller" / "Dockerfile").read_text()
    cmd_line = next(line for line in dockerfile.splitlines() if line.startswith("CMD"))

    # A shell as PID 1 does not forward SIGTERM to Hypercorn.
    assert json.loads(cmd_line[len("CMD") :]) == [
        "python",
        "-m",
        "hypercorn",
        "--config",
        "python:exaflow.controller.quart.hypercorn_config",
        "-b",
        "0.0.0.0:5000",
        "exaflow/controller/quart/app:app",
    ]
