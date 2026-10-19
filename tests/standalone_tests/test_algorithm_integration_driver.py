from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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


def _load_scaffold():
    module_path = (
        REPO_ROOT
        / ".agents/skills/exaflow-algorithm-scaffold/scripts"
        / "scaffold_algorithms.py"
    )
    spec = importlib.util.spec_from_file_location("algorithm_scaffold", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCAFFOLD = _load_scaffold()


def test_runtime_domain_is_independent_of_federated_family(tmp_path):
    paths = DRIVER.expected_paths(
        tmp_path, "example", "statistics", "statistics", runtime_domain="outliers"
    )
    assert paths["algorithm_module"] == (
        tmp_path / "exaflow/algorithms/exareme3/outliers/example.py"
    )
    assert paths["federated_core"] == (
        tmp_path / "exaflow/algorithms/federated/statistics/example.py"
    )


def test_outlier_family_follows_domain_layout(tmp_path):
    paths = DRIVER.expected_paths(tmp_path, "outlier_report", "outliers", "outliers")
    assert paths["federated_core"] == (
        tmp_path / "exaflow/algorithms/federated/outliers/outlier_report.py"
    )
    assert SCAFFOLD.infer_family_from_algorithm("outlier_report") == "outliers"


def test_existing_nested_wrapper_is_reused(tmp_path):
    wrapper = tmp_path / "exaflow/algorithms/exareme3/linear_model/example.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("")
    paths = DRIVER.expected_paths(tmp_path, "example", "statistics", "statistics")
    assert paths["algorithm_module"] == wrapper
    assert VALIDATOR.runtime_import_path(wrapper, tmp_path) == (
        "exaflow.algorithms.exareme3.linear_model.example"
    )


def test_duplicate_runtime_modules_are_rejected(tmp_path):
    for domain in ["statistics", "outliers"]:
        wrapper = tmp_path / "exaflow/algorithms/exareme3" / domain / "example.py"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("")
    with pytest.raises(ValueError, match="Ambiguous runtime modules"):
        DRIVER.expected_paths(tmp_path, "example", "statistics", "statistics")
    report = []
    VALIDATOR.check_import_and_spec(
        "example", report, repo_root=tmp_path, runtime_catalog_set={"example"}
    )
    assert report[0].severity == "failed"
    assert report[0].check == "algorithm_module_resolution"


def test_explicit_runtime_domain_is_forwarded_to_scaffold(tmp_path):
    gates = DRIVER.build_gate_plan(
        algorithm="example",
        family="statistics",
        subfolder="statistics",
        runtime_domain="outliers",
        repo_root=tmp_path,
        strict=False,
        skip_scaffold=False,
    )
    command = gates[0].command
    assert command[command.index("--runtime-domain") + 1] == "outliers"
    assert all("ruff" not in (gate.command or []) for gate in gates)


def test_scaffold_creates_domain_package_and_preserves_core_family(tmp_path):
    report = []
    SCAFFOLD.run_for_algorithm(
        "example",
        report,
        repo_root=tmp_path,
        dry_run=False,
        family_override="statistics",
        subfolder_override=None,
        runtime_domain="outliers",
        with_federated_core=True,
        with_registration=False,
        with_doc_index=False,
        with_sample_fixture=False,
        runtime_catalog_set=set(),
    )
    package = tmp_path / "exaflow/algorithms/exareme3/outliers"
    assert (package / "__init__.py").exists()
    assert (package / "example.py").exists()
    assert (tmp_path / "exaflow/algorithms/federated/statistics/example.py").exists()
    assert not (tmp_path / "exaflow/algorithms/exareme3/example.py").exists()
    prod_test = (tmp_path / "tests/prod_env_tests/test_example.py").read_text()
    assert "algorithm_layout" not in prod_test
    assert "sys.path" not in prod_test


def test_validator_detects_nested_changes_without_selecting_helpers():
    targets = VALIDATOR.map_changed_files_to_algorithms(
        [
            "exaflow/algorithms/exareme3/outliers/outlier_report.py",
            "exaflow/algorithms/exareme3/linear_model/linear_regression_cv.py",
            "exaflow/algorithms/exareme3/utils/preprocessing_step.py",
            "exaflow/algorithms/exareme3/outliers/__init__.py",
        ]
    )
    assert targets == {"outlier_report", "linear_regression_cv"}


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
