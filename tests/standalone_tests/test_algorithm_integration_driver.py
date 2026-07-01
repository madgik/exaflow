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


def _load_validator():
    module_path = (
        REPO_ROOT
        / ".agents"
        / "skills"
        / "exaflow-algorithm-validate"
        / "scripts"
        / "validate_algorithms.py"
    )
    spec = importlib.util.spec_from_file_location(
        "exaflow_algorithm_validator",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


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


def test_validator_detects_fixture_replacement_placeholders():
    tokens = VALIDATOR.find_placeholder_tokens(
        '{"data_model": "__REPLACE_ME_DATA_MODEL__", "note": "TODO"}'
    )

    assert tokens == ["TODO", "__REPLACE_ME_DATA_MODEL__"]


def test_validator_fails_fixture_with_replacement_placeholders(tmp_path):
    fixture_path = tmp_path / "expected.json"
    fixture_path.write_text(
        """
        {
          "test_cases": [
            {
              "input": {
                "inputdata": {
                  "y": ["__REPLACE_ME_Y__"],
                  "x": ["group"],
                  "data_model": "dementia:0.1",
                  "datasets": ["ppmi0"],
                  "filters": null
                },
                "parameters": {}
              },
              "output": {}
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    report = []

    VALIDATOR.check_fixture_content(
        "example_test",
        report,
        repo_root=tmp_path,
        fixture_path=fixture_path,
        require_non_empty=True,
    )

    rows = [entry.to_dict() for entry in report]
    placeholder_rows = [
        row for row in rows if row["check"] == "prod_env_expected_placeholder_check"
    ]
    assert placeholder_rows
    assert placeholder_rows[0]["severity"] == "failed"


def test_validator_algorithm_profile_allows_algorithm_and_registration_paths():
    changed_files = [
        "exaflow/algorithms/exareme3/example_test.py",
        "exaflow/algorithms/federated/statistics/example_test.py",
        "exaflow/algorithms/federated/statistics/__init__.py",
        "exaflow/algorithms/federated/__init__.py",
        "exaflow/algorithms/federated/README.md",
        "exaflow/algorithms/specifications.py",
        "tests/standalone_tests/federated_algorithms/statistics/test_example_test.py",
        "tests/prod_env_tests/test_example_test.py",
        "tests/prod_env_tests/expected/example_test_expected.json",
        "documentation/algorithms/example_test.md",
    ]

    violations = VALIDATOR.algorithm_profile_boundary_violations(changed_files)

    assert violations == []


def test_validator_algorithm_profile_blocks_system_owned_paths():
    changed_files = [
        "exaflow/algorithms/exareme3/example_test.py",
        "exaflow/controller/quart/endpoints.py",
        "exaflow/worker/grpc_server.py",
        "exaflow/protos/worker/worker.proto",
        "kubernetes/values.yaml",
        "pyproject.toml",
    ]

    violations = VALIDATOR.algorithm_profile_boundary_violations(changed_files)

    assert violations == [
        "exaflow/controller/quart/endpoints.py",
        "exaflow/protos/worker/worker.proto",
        "exaflow/worker/grpc_server.py",
        "kubernetes/values.yaml",
        "pyproject.toml",
    ]


def test_validator_reports_algorithm_profile_boundary_failure(tmp_path):
    report = []

    VALIDATOR.check_algorithm_profile_boundary(
        report,
        repo_root=tmp_path,
        changed_files=[
            "exaflow/algorithms/exareme3/example_test.py",
            "exaflow/controller/services/api/algorithm_request_dtos.py",
        ],
    )

    rows = [entry.to_dict() for entry in report]
    failed = [row for row in rows if row["severity"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["check"] == "algorithm_profile_boundary"
    assert "System Feature Request" in failed[0]["next_action"]
