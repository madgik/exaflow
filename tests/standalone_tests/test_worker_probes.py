import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")
@pytest.mark.parametrize("smpc", ["true", "false"])
def test_worker_deployment_configuration(smpc):
    rendered = subprocess.check_output(
        [
            "helm",
            "template",
            str(REPO_ROOT / "kubernetes"),
            "--set",
            f"smpc.enabled={smpc}",
            "--set",
            "worker_count=3",
        ],
        text=True,
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc]
    assert not any("globalworker" in doc["metadata"]["name"] for doc in docs)
    worker_set = next(
        doc
        for doc in docs
        if doc["kind"] == "StatefulSet"
        and doc["metadata"]["name"] == "exaflow-localworker"
    )
    assert worker_set["spec"]["replicas"] == 3
    containers = worker_set["spec"]["template"]["spec"]["containers"]
    worker = next(
        container for container in containers if container["name"] == "worker"
    )
    env = {entry["name"]: entry.get("value") for entry in worker["env"]}
    assert "WORKER_ROLE" not in env
    assert env["GRPC_MAX_WORKERS"] == "36"
    assert ("smpc-client" in {c["name"] for c in containers}) == (smpc == "true")
    for probe in ("startupProbe", "livenessProbe", "readinessProbe"):
        assert worker[probe]["timeoutSeconds"] == 30
