import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from exaflow.controller.http import server_config as hc
from exaflow.controller.services import controller_shutdown as budget

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


def _prestop_delay(block: str) -> int:
    return int(re.search(r'command: \["sleep", "(\d+)"\]', block).group(1))


def _shutdown_budget_seconds(prestop_delay: int) -> int:
    """Pod grace must cover the real shutdown path, not worker_tasks_timeout.

    The preStop delay runs first. After SIGTERM Uvicorn drains for
    GRACEFUL_TIMEOUT, then the lifespan hook cancels the analysis RPCs
    (SHUTDOWN_RPC_CANCEL_TIMEOUT) and aggregation cleanup is bounded by
    SHUTDOWN_CLEANUP_TIMEOUT, instead of waiting out worker_tasks_timeout. The
    hook has SHUTDOWN_TIMEOUT.
    """
    worker_tasks_timeout = 300
    assert budget.SHUTDOWN_RPC_CANCEL_TIMEOUT + budget.SHUTDOWN_CLEANUP_TIMEOUT < (
        worker_tasks_timeout
    )
    assert budget.SHUTDOWN_CLEANUP_TIMEOUT <= budget.SHUTDOWN_TIMEOUT
    return int(
        prestop_delay
        + budget.GRACEFUL_TIMEOUT
        + budget.SHUTDOWN_RPC_CANCEL_TIMEOUT
        + budget.SHUTDOWN_CLEANUP_TIMEOUT
        + budget.SHUTDOWN_TIMEOUT
        + budget.SHUTDOWN_GRACE_MARGIN
    )


def test_uvicorn_graceful_timeout_is_120_seconds():
    assert hc.graceful_timeout == 120 == budget.GRACEFUL_TIMEOUT
    assert budget.SHUTDOWN_TIMEOUT == 60


def test_uvicorn_serves_in_process():
    """workers > 1 puts Uvicorn's supervisor in front of the serving process.

    Kubernetes signals PID 1 and `inv kill-controller` signals the process that
    listens. Both must be the one that serves.
    """
    assert hc.workers == 1


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_controller_pod_grace_period_covers_uvicorn_shutdown():
    grace, block = _helm_grace()
    timeout_at = block.index("name: WORKER_TASKS_TIMEOUT")
    worker_tasks_timeout = int(
        re.search(r'value: "(\d+)"', block[timeout_at : timeout_at + 80]).group(1)
    )

    expected = _shutdown_budget_seconds(_prestop_delay(block))
    # The rendered grace follows the bounded path. Adding worker_tasks_timeout
    # on top would mean shutdown still waits out a worker RPC.
    assert worker_tasks_timeout == 300
    assert grace == expected
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
def test_controller_pod_keeps_serving_while_endpoints_update():
    """Uvicorn closes its listen socket at SIGTERM. A preStop sleep keeps the
    pod serving until Endpoints stop routing to it, and the grace period
    counts it."""
    grace, block = _helm_grace()
    overridden, overridden_block = _helm_grace(
        "--set", "controller.shutdown_prestop_delay=12"
    )

    assert _prestop_delay(block) == 5
    assert _prestop_delay(overridden_block) == 12
    assert overridden == grace - 5 + 12


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_liveness_probe_does_not_set_its_own_grace_period():
    """A probe-level terminationGracePeriodSeconds would SIGKILL on its own
    clock. Kubelet stops liveness probes of a terminating pod, so /livez
    failing because the listen socket is closed must not restart the pod.
    """
    _grace, block = _helm_grace()
    start = block.index("livenessProbe:")
    end = block.index("readinessProbe:")
    assert "terminationGracePeriodSeconds" not in block[start:end]


def test_controller_image_command_is_exec_form_without_shell():
    dockerfile = (REPO_ROOT / "exaflow" / "controller" / "Dockerfile").read_text()
    cmd_line = next(line for line in dockerfile.splitlines() if line.startswith("CMD"))

    # A shell as PID 1 does not forward SIGTERM to Uvicorn.
    assert json.loads(cmd_line[len("CMD") :]) == [
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
