from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_driver():
    module_path = (
        REPO_ROOT
        / ".agents"
        / "skills"
        / "exaflow-algorithm-scaffold"
        / "scripts"
        / "integrate_new_algorithm.py"
    )
    spec = importlib.util.spec_from_file_location(
        "exaflow_algorithm_integration_driver",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DRIVER = _load_driver()


def test_extract_json_object_handles_poetry_preamble():
    payload = DRIVER.extract_json_object(
        'Using python3.10 (3.10.20)\n{"failed": [], "warnings": []}\n'
    )

    assert payload == {"failed": [], "warnings": []}


def test_build_gate_plan_includes_mechanical_definition_of_done_gates(tmp_path):
    algorithm = "example_test"
    family = "statistics"
    paths = DRIVER.expected_paths(tmp_path, algorithm, family, family)

    for path in [
        paths["algorithm_module"],
        paths["federated_core"],
        paths["standalone_test"],
        paths["prod_test"],
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n", encoding="utf-8")

    gates = DRIVER.build_gate_plan(
        algorithm=algorithm,
        family=family,
        subfolder=family,
        repo_root=tmp_path,
        strict=True,
        skip_scaffold=False,
    )

    gate_names = [gate.name for gate in gates]
    assert gate_names == [
        "scaffold",
        "validate_new_algorithm",
        "ruff_import_order",
        "ruff_format",
        "standalone_pytest",
    ]
    validate_gate = next(
        gate for gate in gates if gate.name == "validate_new_algorithm"
    )
    assert "--strict" in validate_gate.command


def test_build_gate_plan_marks_missing_file_gates_as_non_runnable(tmp_path):
    gates = DRIVER.build_gate_plan(
        algorithm="missing_test",
        family="statistics",
        subfolder="statistics",
        repo_root=tmp_path,
        strict=False,
        skip_scaffold=True,
    )

    assert gates[0].name == "validate_new_algorithm"
    assert any(gate.name == "ruff_checks" and gate.command is None for gate in gates)
    assert any(
        gate.name == "standalone_pytest" and gate.command is None for gate in gates
    )


def test_summarize_requires_all_gates_to_pass():
    passed = DRIVER.GateResult(
        name="scaffold",
        status="pass",
        severity="pass",
        command=["true"],
        elapsed_seconds=0.1,
        returncode=0,
        message="Gate passed.",
        next_action=None,
    )
    failed = DRIVER.GateResult(
        name="validate_new_algorithm",
        status="failed",
        severity="failed",
        command=["false"],
        elapsed_seconds=0.2,
        returncode=1,
        message="Validator reported 1 failed rows.",
        next_action="Fix validator output.",
    )

    summary = DRIVER.summarize(
        algorithm="example_test",
        family="statistics",
        strict=False,
        gates=[passed, failed],
        elapsed_seconds=0.3,
    )

    assert summary["done"] is False
    assert len(summary["passed"]) == 1
    assert len(summary["failed"]) == 1
