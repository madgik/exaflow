"""Integration-tool regressions for algorithms, preprocessing and helper modules."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / ".agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py"
)
SPEC = importlib.util.spec_from_file_location("component_validator_tests", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def _write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_preprocessing_catalog_is_distinct_from_analysis_catalog(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "exaflow",
        SimpleNamespace(
            exareme3_algorithm_classes={"pca": object()},
            exareme3_preprocessing_step_classes={"variable_scaler": object()},
        ),
    )
    assert VALIDATOR.load_runtime_catalog(REPO_ROOT) == ["pca"]
    assert VALIDATOR.load_runtime_catalog(REPO_ROOT, preprocessing=True) == [
        "variable_scaler"
    ]


def test_changed_selection_does_not_invent_algorithms_for_helpers(
    tmp_path, monkeypatch, capsys
):
    _write(tmp_path, "pyproject.toml", "")
    _write(
        tmp_path, "exaflow/algorithms/exareme3/transformations/variable_scaler.py", ""
    )
    changed = [
        "exaflow/algorithms/exareme3/transformations/variable_scaler.py",
        "exaflow/algorithms/federated/feature_engineering/state.py",
        "tests/standalone_tests/federated_algorithms/feature_engineering/test_scalers.py",
        "tests/prod_env_tests/test_derived_transformations.py",
    ]
    monkeypatch.setattr(
        VALIDATOR,
        "parse_args",
        lambda: argparse.Namespace(
            repo_root=str(tmp_path),
            algorithms=None,
            new_algorithm=None,
            changed_only=False,
            all=False,
            strict=False,
        ),
    )
    monkeypatch.setattr(
        VALIDATOR,
        "load_runtime_catalog",
        lambda root, **kwargs: (
            ["variable_scaler"] if kwargs.get("preprocessing") else ["pca"]
        ),
    )
    monkeypatch.setattr(VALIDATOR, "get_changed_files", lambda root: changed)
    monkeypatch.setattr(VALIDATOR, "run_fast_tier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        VALIDATOR, "check_import_and_spec", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        VALIDATOR, "check_changed_source_imports", lambda *args, **kwargs: None
    )
    VALIDATOR.main()
    result = json.loads(capsys.readouterr().out)
    assert result["targets"] == ["variable_scaler"]


def test_preprocessing_probe_uses_its_own_base_class(monkeypatch):
    def probe(command, **kwargs):
        code = command[-1]
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "classes_found": 1 if "import PreprocessingStep" in code else 0,
                    "match": "import PreprocessingStep" in code,
                    "valid_specs": True,
                }
            ),
            "",
        )

    monkeypatch.setattr(VALIDATOR.subprocess, "run", probe)
    report = []
    VALIDATOR.check_import_and_spec(
        "variable_scaler",
        report,
        repo_root=REPO_ROOT,
        runtime_catalog_set={"variable_scaler"},
    )
    assert not [row for row in report if row.severity == "failed"]


def test_grouped_preprocessing_tests_are_real_coverage(tmp_path):
    module = "exaflow.algorithms.exareme3.transformations.variable_scaler"
    _write(
        tmp_path,
        "exaflow/algorithms/exareme3/transformations/variable_scaler.py",
        "from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep\nclass Scaler(PreprocessingStep): pass\n",
    )
    test = _write(
        tmp_path,
        "tests/standalone_tests/exareme3/transformations/test_contract.py",
        f"from {module} import Scaler\ndef test_scaler(): assert Scaler\n",
    )
    assert VALIDATOR.discover_standalone_paths(tmp_path, "variable_scaler") == [test]


def test_removed_package_does_not_require_restoring_old_initializer(tmp_path):
    report = []
    VALIDATOR.check_touched_registration_files(
        report,
        algorithms=[],
        repo_root=tmp_path,
        changed_files=["exaflow/algorithms/federated/preprocessing/__init__.py"],
    )
    assert report and all(row.severity == "pass" for row in report)


def test_missing_initializer_in_live_package_is_still_an_error(tmp_path):
    _write(
        tmp_path,
        "exaflow/algorithms/federated/feature_engineering/scaler.py",
        "VALUE = 1\n",
    )
    report = []
    VALIDATOR.check_touched_registration_files(
        report,
        algorithms=[],
        repo_root=tmp_path,
        changed_files=["exaflow/algorithms/federated/feature_engineering/__init__.py"],
    )
    assert any(row.severity == "failed" for row in report)


@pytest.mark.parametrize("failure", ["registration", "import", "name", "outputs"])
def test_invalid_preprocessing_is_not_accepted(monkeypatch, failure):
    def probe(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1 if failure == "import" else 0,
            json.dumps(
                {
                    "classes_found": 1,
                    "match": failure != "name",
                    "valid_specs": failure != "outputs",
                }
            ),
            "broken import" if failure == "import" else "",
        )

    monkeypatch.setattr(VALIDATOR.subprocess, "run", probe)
    report = []
    VALIDATOR.check_import_and_spec(
        "variable_scaler",
        report,
        repo_root=REPO_ROOT,
        runtime_catalog_set=set() if failure == "registration" else {"variable_scaler"},
    )
    assert any(row.severity == "failed" for row in report)


def test_analysis_algorithm_still_requires_expected_fixture(tmp_path):
    _write(
        tmp_path,
        "tests/standalone_tests/federated_algorithms/statistics/test_example.py",
        "def test_example(): pass\n",
    )
    _write(
        tmp_path, "tests/prod_env_tests/test_example.py", "def test_example(): pass\n"
    )
    _write(tmp_path, "documentation/algorithms/example.md", "Example\n")
    report = []
    VALIDATOR.check_required_paths(
        "example", report, repo_root=tmp_path, enforce_canonical=False
    )
    assert any(
        row.check == "prod_env_expected_exists" and row.severity == "failed"
        for row in report
    )


def _step_module(root):
    return _write(
        root,
        "exaflow/algorithms/exareme3/transformations/example_step.py",
        "from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep\nclass Example(PreprocessingStep): pass\n",
    )


def test_grouped_preprocessing_contract_does_not_require_fake_analysis_fixture(
    tmp_path,
):
    _step_module(tmp_path)
    _write(
        tmp_path,
        "tests/standalone_tests/exareme3/transformations/test_grouped.py",
        'STEP = "example_step"\ndef test_contract(): pass\n',
    )
    _write(
        tmp_path,
        "tests/prod_env_tests/test_grouped.py",
        'STEP = "example_step"\ndef test_http(): pass\n',
    )
    _write(tmp_path, "documentation/algorithms/example_step.md", "Example\n")
    report = []
    paths = VALIDATOR.check_required_paths(
        "example_step", report, repo_root=tmp_path, enforce_canonical=False
    )
    assert all(row.severity == "pass" for row in report)
    assert paths.prod_expected is None
    assert paths.standalone.name == "test_grouped.py"


def test_unrelated_tests_and_comments_do_not_count_as_step_coverage(tmp_path):
    _step_module(tmp_path)
    _write(
        tmp_path,
        "tests/standalone_tests/exareme3/test_other.py",
        "# example_step\ndef test_other(): pass\n",
    )
    _write(tmp_path, "tests/prod_env_tests/test_other.py", 'STEP = "different_step"\n')
    report = []
    VALIDATOR.check_required_paths(
        "example_step", report, repo_root=tmp_path, enforce_canonical=False
    )
    assert {row.check for row in report if row.severity == "failed"} == {
        "standalone_test_exists",
        "prod_env_test_exists",
        "documentation_exists",
    }


def test_strict_tier_executes_all_referencing_grouped_tests(tmp_path, monkeypatch):
    _step_module(tmp_path)
    for name in ("first", "second"):
        _write(
            tmp_path, f"tests/prod_env_tests/test_{name}.py", 'STEP = "example_step"\n'
        )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return 0, "2 passed", ""

    monkeypatch.setattr(VALIDATOR, "run_command", run)
    report = []
    VALIDATOR.run_strict_tier(report, algorithms=["example_step"], repo_root=tmp_path)
    assert commands[0][-2:] == [
        "tests/prod_env_tests/test_first.py",
        "tests/prod_env_tests/test_second.py",
    ]
    assert all(row.severity == "pass" for row in report)


@pytest.mark.parametrize("error", [False, True])
def test_helpers_have_import_checks_without_algorithm_requirements(
    tmp_path, monkeypatch, error
):
    source = "exaflow/algorithms/federated/feature_engineering/state.py"
    _write(tmp_path, source, "VALUE = 1\n")

    def run(command, **kwargs):
        assert "feature_engineering.state" in command[-1]
        return 0, json.dumps({"state": "ModuleNotFoundError"} if error else {}), ""

    monkeypatch.setattr(VALIDATOR, "run_command", run)
    report = []
    VALIDATOR.check_changed_source_imports(
        report, repo_root=tmp_path, changed_files=[source]
    )
    assert report[0].severity == ("failed" if error else "pass")


@pytest.mark.parametrize("returncode", [0, 1])
def test_changed_core_tests_run_even_without_runtime_id(
    tmp_path, monkeypatch, returncode
):
    source = (
        "tests/standalone_tests/federated_algorithms/feature_engineering/test_state.py"
    )
    _write(tmp_path, source, "def test_state(): pass\n")

    def run(command, **kwargs):
        assert source in command
        return (
            returncode,
            "a useful assertion failure" if returncode else "1 passed",
            "",
        )

    monkeypatch.setattr(VALIDATOR, "run_command", run)
    report = []
    VALIDATOR.run_fast_tier(
        report, algorithms=[], repo_root=tmp_path, changed_files=[source]
    )
    assert report[0].severity == ("failed" if returncode else "pass")
    if returncode:
        assert "assertion failure" in report[0].message


def test_runtime_helper_is_not_mistaken_for_public_component(tmp_path):
    _write(
        tmp_path,
        "exaflow/algorithms/exareme3/transformations/formatting.py",
        "def format_value(): pass\n",
    )
    paths = ["exaflow/algorithms/exareme3/transformations/formatting.py"]
    assert VALIDATOR.filter_runtime_targets(tmp_path, ["formatting"], paths) == []


def test_deleted_runtime_wrapper_is_still_checked(tmp_path):
    paths = ["exaflow/algorithms/exareme3/transformations/example_step.py"]
    assert VALIDATOR.filter_runtime_targets(tmp_path, ["example_step"], paths) == [
        "example_step"
    ]


@pytest.mark.parametrize("exported", [True, False])
def test_actual_federated_class_names_are_checked_without_invented_aliases(
    tmp_path, exported
):
    _write(
        tmp_path,
        "exaflow/algorithms/federated/decomposition/pca.py",
        "class FederatedPCA: pass\n",
    )
    initializers = [
        "exaflow/algorithms/federated/__init__.py",
        "exaflow/algorithms/federated/decomposition/__init__.py",
    ]
    for path in initializers:
        _write(tmp_path, path, "from .pca import FederatedPCA\n" if exported else "")
    report = []
    VALIDATOR.check_touched_registration_files(
        report, algorithms=["pca"], repo_root=tmp_path, changed_files=initializers
    )
    assert any(row.severity == "failed" for row in report) is not exported
