import ast
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import Path

import pytest

FEDERATED_PACKAGE = "exaflow.algorithms.federated"


def _external_exaflow_imports(source: str, package: str) -> Iterator[tuple[int, str]]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            try:
                targets = [resolve_name(name, package)]
            except ImportError:
                yield node.lineno, name
                continue
        else:
            continue

        for target in targets:
            if target.partition(".")[0] == "exaflow" and not (
                target == FEDERATED_PACKAGE
                or target.startswith(FEDERATED_PACKAGE + ".")
            ):
                yield node.lineno, target


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import exaflow.algorithms.federated as federated", []),
        ("from exaflow.algorithms.federated.exceptions import FedValueError", []),
        ("from . import describe\nfrom ..exceptions import FedValueError", []),
        ("import math, numpy, exaflow_helpers", []),
        ("import numpy, exaflow.worker as worker", [(1, "exaflow.worker")]),
        ("from exaflow import column_names", [(1, "exaflow")]),
        ("from exaflow.algorithms import federated", [(1, "exaflow.algorithms")]),
        (
            "import exaflow.algorithms.federated_extra",
            [(1, "exaflow.algorithms.federated_extra")],
        ),
        ("from ... import specifications", [(1, "exaflow.algorithms")]),
        ("from ....column_names import DATASET_COL", [(1, "exaflow.column_names")]),
        ("from .....column_names import DATASET_COL", [(1, ".....column_names")]),
        (
            "if TYPE_CHECKING:\n    from exaflow.worker import config\n"
            "def helper():\n    import exaflow.algorithms.exareme3",
            [(2, "exaflow.worker"), (4, "exaflow.algorithms.exareme3")],
        ),
    ],
)
def test_federated_import_scanner(source, expected):
    package = FEDERATED_PACKAGE + ".statistics"
    assert list(_external_exaflow_imports(source, package)) == expected


def test_federated_package_has_no_external_exaflow_imports():
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root.joinpath(*FEDERATED_PACKAGE.split("."))
    paths = sorted(package_root.rglob("*.py"))
    assert paths, f"No Python files found under {package_root}"

    violations = []
    for path in paths:
        relative_path = path.relative_to(repo_root)
        package = ".".join(relative_path.parent.parts)
        imports = _external_exaflow_imports(path.read_text(encoding="utf-8"), package)
        violations.extend(
            f"{relative_path}:{line}: {target}" for line, target in sorted(imports)
        )

    assert not violations, "Imports outside the federated package:\n" + "\n".join(
        violations
    )
