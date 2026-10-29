import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE_FIELDS = ("periodSeconds", "timeoutSeconds", "failureThreshold")


def _worker_containers() -> dict:
    rendered = subprocess.check_output(
        ["helm", "template", str(REPO_ROOT / "kubernetes")],
        text=True,
    )
    containers = {}
    for doc in yaml.safe_load_all(rendered):
        if not doc or doc.get("kind") != "StatefulSet":
            continue
        name = doc["metadata"]["name"]
        for container in doc["spec"]["template"]["spec"]["containers"]:
            if container["name"] == "worker":
                containers[name] = container
    return containers


def _env(container: dict, name: str) -> str:
    return next(var["value"] for var in container["env"] if var["name"] == name)


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_globalworker_probes_tolerate_slow_healthchecks_like_localworkers():
    # The healthcheck spawns a Python process; on a busy node it can take
    # longer than 10s, and three misses restarted the globalworker mid-run.
    containers = _worker_containers()
    globalworker = containers["exaflow-globalworker"]
    localworker = containers["exaflow-localworker"]

    def fields(container, probe):
        return {key: container[probe].get(key) for key in PROBE_FIELDS}

    assert fields(globalworker, "livenessProbe") == fields(localworker, "livenessProbe")
    assert fields(globalworker, "readinessProbe") == fields(
        localworker, "readinessProbe"
    )
    for probe in ("startupProbe", "livenessProbe", "readinessProbe"):
        assert globalworker[probe]["timeoutSeconds"] == 30, probe


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
def test_globalworker_grpc_pool_scales_with_concurrent_experiments():
    containers = _worker_containers()

    assert _env(containers["exaflow-globalworker"], "GRPC_MAX_WORKERS") == _env(
        containers["exaflow-localworker"], "GRPC_MAX_WORKERS"
    )
