import re
import subprocess
import sys
from pathlib import Path

from exaflow.aggregation_clients import AggregationType as ClientAggregationType
from exaflow.aggregation_types import AggregationType


def test_clients_and_server_share_one_aggregation_type_enum():
    from exaflow.aggregation_server import server

    assert ClientAggregationType is AggregationType
    assert server.AggregationType is AggregationType
    assert set(server.REDUCERS) == {
        AggregationType.SUM,
        AggregationType.MIN,
        AggregationType.MAX,
    }


def test_aggregation_type_compares_equal_to_its_wire_string():
    assert AggregationType.UNION == "UNION"
    assert AggregationType("SUM") is AggregationType.SUM
    assert str(AggregationType.MAX) == "MAX"


def test_server_imports_without_the_client_package(tmp_path):
    # The aggregation server image copies an explicit list of modules from
    # exaflow/ (see exaflow/aggregation_server/Dockerfile) without
    # exaflow/__init__.py, making exaflow a namespace package inside the image.
    # Mirror that exact COPY set into tmp_path and verify the server imports
    # cleanly without exaflow.aggregation_clients or any uncopied module.
    repo_root = Path(__file__).resolve().parents[3]
    dockerfile = (repo_root / "exaflow/aggregation_server/Dockerfile").read_text()
    copied = re.findall(r"^COPY\s+(exaflow/\S+)\s+\./(exaflow/\S+)", dockerfile, re.M)
    assert copied, "Expected COPY exaflow/... lines in aggregation_server/Dockerfile"

    for src_rel, dst_rel in copied:
        dst = tmp_path / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(repo_root / src_rel)

    code = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r}); "
        "import exaflow.aggregation_server.server; "
        "assert 'exaflow.aggregation_clients' not in sys.modules, "
        "'server imports the client package'"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)
