"""Keep FETA's implementation independent from legacy and shared algorithm code."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FETA_PATHS = (
    "exaflow/algorithms/feta",
    "exaflow/feta",
    "exaflow/controller/services/feta",
    "exaflow/worker/feta",
)
FETA_MODULES = tuple(path.replace("/", ".") for path in FETA_PATHS) + (
    "exaflow.protos.feta",
)
# Public engine integration contracts; implementation helpers must stay in FETA.
PLATFORM_IMPORTS = {
    "exaflow": {"feta_algorithm_classes"},
    "exaflow.algorithms": {"specifications"},
    "exaflow.algorithms.specifications": None,
    "exaflow.controller": {"config"},
    "exaflow.controller.services": {"get_worker_landscape_aggregator"},
    "exaflow.controller.services.strategy_interface": {"AlgorithmExecutionStrategyI"},
    "exaflow.worker": {"config"},
    "exaflow.worker_communication": {"BadUserInput", "InsufficientDataError"},
}


def test_feta_imports_only_owned_code_and_public_platform_contracts():
    paths = [file for folder in FETA_PATHS for file in (ROOT / folder).rglob("*.py")]
    paths.append(ROOT / "exaflow/aggregation_server/feta.py")
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    continue  # Relative imports are within the owning FETA package.
                imports = [(node.module, [alias.name for alias in node.names])]
            elif isinstance(node, ast.Import):
                imports = [(alias.name, []) for alias in node.names]
            else:
                continue
            for module, names in imports:
                if not module or not module.startswith("exaflow"):
                    continue
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in FETA_MODULES
                ):
                    continue
                assert module in PLATFORM_IMPORTS, (path.relative_to(ROOT), module)
                allowed = PLATFORM_IMPORTS[module]
                if allowed is not None:
                    assert set(names) <= allowed, (
                        path.relative_to(ROOT),
                        module,
                        names,
                    )
