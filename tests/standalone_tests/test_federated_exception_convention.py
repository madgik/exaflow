import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

PACKAGE = "exaflow.algorithms.federated"
FED_ERROR = PACKAGE + ".exceptions.FedError"


def _failure_violations(sources: dict[str, str]) -> list[str]:
    """Check explicit raises; constructors own runtime reason validation."""
    modules = {}
    bases = {}
    for module, source in sources.items():
        tree = ast.parse(source)
        namespace = module.removesuffix(".__init__")
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = resolve_name(
                        "." * node.level + target, module.rpartition(".")[0]
                    )
                for alias in node.names:
                    imports[alias.asname or alias.name] = target + "." + alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.partition(".")[0]] = (
                        alias.name if alias.asname else alias.name.partition(".")[0]
                    )

        def qualified(node, imports=imports, namespace=namespace):
            name = ast.unparse(node)
            first, separator, rest = name.partition(".")
            return imports.get(first, namespace + "." + first) + (
                separator + rest if separator else ""
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases[namespace + "." + node.name] = {
                    qualified(base) for base in node.bases
                }
        modules[module] = tree, qualified

    family = {FED_ERROR}
    while True:
        extended = family | {
            name for name, parents in bases.items() if parents & family
        }
        if extended == family:
            break
        family = extended

    violations = []
    for module, (tree, qualified) in modules.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            call = node.exc
            if not isinstance(call, ast.Call) or qualified(call.func) not in family:
                problem = "raise a FedError subclass directly"
            elif not call.args and not any(k.arg == "reason" for k in call.keywords):
                problem = "supply a reason"
            else:
                continue
            violations.append(f"{module}:{node.lineno}: {problem}")
    return violations


@pytest.mark.parametrize(
    "source, valid",
    [
        (
            "from exaflow.algorithms.federated.exceptions import FedError as E\n"
            "class SpecificError(E): pass\n"
            "raise SpecificError(reason=reason)",
            True,
        ),
        ("raise ValueError('missing metadata')", False),
        ("from builtins import TypeError as E\nraise E('bad type')", False),
        (
            "from exaflow.algorithms.federated.exceptions import FedError\n"
            "raise FedError()",
            False,
        ),
        ("raise error", False),
        ("try:\n    operation()\nexcept Exception:\n    raise", True),
    ],
)
def test_failure_scanner(source, valid):
    assert (not _failure_violations({PACKAGE + ".example": source})) is valid


def test_federated_failures_require_library_exceptions_and_reasons():
    root = Path(__file__).resolve().parents[2] / "exaflow/algorithms/federated"
    sources = {
        PACKAGE
        + "."
        + ".".join(path.relative_to(root).with_suffix("").parts): path.read_text(
            encoding="utf-8"
        )
        for path in sorted(root.rglob("*.py"))
    }
    assert sources
    violations = _failure_violations(sources)
    assert not violations, "Federated exception convention violations:\n" + "\n".join(
        violations
    )
