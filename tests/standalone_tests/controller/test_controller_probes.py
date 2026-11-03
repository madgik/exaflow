import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from hypercorn.config import Config

from exaflow.controller.quart import hypercorn_config as hc

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


def _helm_grace(*extra: str) -> tuple[int, str]:
    rendered = subprocess.check_output(
        ["helm", "template", str(REPO_ROOT / "kubernetes"), *extra],
        text=True,
    )
    block = _controller_deployment_block(rendered)
    grace = int(re.search(r"terminationGracePeriodSeconds: (\d+)", block).group(1))
    return grace, block


def _shutdown_budget_seconds() -> int:
    """Pod grace must cover the real shutdown path, not worker_tasks_timeout.

    After graceful_timeout Hypercorn cancels in-flight requests and then waits
    for them to finish. Shutdown cancels those RPCs (shutdown_rpc_cancel_timeout)
    and bounds aggregation cleanup (shutdown_cleanup_timeout) instead of waiting
    out worker_tasks_timeout. A call started near the end of the drain would
    otherwise still be running near t=420. after_serving then has shutdown_timeout.
    """
    worker_tasks_timeout = 300
    assert hc.shutdown_rpc_cancel_timeout + hc.shutdown_cleanup_timeout < (
        worker_tasks_timeout
    )
    assert hc.shutdown_cleanup_timeout <= hc.shutdown_timeout
    return int(
        hc.graceful_timeout
        + hc.shutdown_rpc_cancel_timeout
        + hc.shutdown_cleanup_timeout
        + hc.shutdown_timeout
        + hc.shutdown_grace_margin
    )


def test_hypercorn_graceful_timeout_is_120_seconds():
    config = Config.from_object("exaflow.controller.quart.hypercorn_config")
    assert config.graceful_timeout == 120
    assert config.shutdown_timeout == 60


def test_hypercorn_serves_in_process_so_the_app_observes_sigterm():
    """workers >= 1 puts Hypercorn's supervisor in front of the app process.

    Kubernetes signals PID 1. The app has to be that process, otherwise a
    shutdown flag installed in the worker never sees SIGTERM and in-flight
    RPCs keep running until worker_tasks_timeout.
    """
    config = Config.from_object("exaflow.controller.quart.hypercorn_config")
    assert hc.workers == 0
    assert config.workers == 0


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_controller_pod_grace_period_covers_hypercorn_shutdown():
    grace, block = _helm_grace()
    timeout_at = block.index("name: WORKER_TASKS_TIMEOUT")
    worker_tasks_timeout = int(
        re.search(r'value: "(\d+)"', block[timeout_at : timeout_at + 80]).group(1)
    )

    budget = _shutdown_budget_seconds()
    # The rendered grace follows the bounded path. Adding worker_tasks_timeout
    # on top would mean shutdown still waits out a worker RPC that ends near
    # t=420, plus cleanup.
    assert worker_tasks_timeout == 300
    assert grace == budget
    assert grace < worker_tasks_timeout + 120


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_controller_pod_grace_tracks_the_cleanup_bound_override():
    default_grace, _block = _helm_grace()
    overridden, _block = _helm_grace("--set", "controller.shutdown_cleanup_timeout=45")
    short_worker_timeout, _block = _helm_grace(
        "--set", "controller.worker_tasks_timeout=30"
    )

    assert overridden == default_grace - 10 + 45
    assert short_worker_timeout == default_grace


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_liveness_probe_does_not_set_its_own_grace_period():
    """A probe-level terminationGracePeriodSeconds would SIGKILL on its own
    clock. Kubelet 1.25+ stops liveness before it sends SIGTERM, so /livez
    failing because the listen socket is closed must not restart the pod.
    """
    _grace, block = _helm_grace()
    start = block.index("livenessProbe:")
    end = block.index("readinessProbe:")
    assert "terminationGracePeriodSeconds" not in block[start:end]


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
        "exaflow.controller.http.app:app",
    ]
