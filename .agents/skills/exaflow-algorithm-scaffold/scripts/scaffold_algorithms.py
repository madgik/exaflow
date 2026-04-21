#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPORT_FIELDS = ("algorithm", "phase", "check", "status", "message", "path")

STANDALONE_FOLDER_RULES = {
    "cluster": {"kmeans"},
    "decomposition": {"pca", "pca_with_transformation"},
    "linear_model": {
        "linear_regression",
        "linear_regression_cv",
        "logistic_regression",
        "logistic_regression_cv",
        "linear_svm",
    },
    "naive_bayes": {
        "naive_bayes_gaussian",
        "naive_bayes_gaussian_cv",
        "naive_bayes_categorical",
        "naive_bayes_categorical_cv",
    },
    "statistics": {
        "anova_oneway",
        "anova_twoway",
        "describe",
        "histogram",
        "pearson_correlation",
        "ttest_independent",
        "ttest_onesample",
        "ttest_paired",
    },
}


@dataclass
class ReportEntry:
    algorithm: str
    phase: str
    check: str
    status: str
    message: str
    path: str | None

    def to_dict(self) -> dict:
        return {field: getattr(self, field) for field in REPORT_FIELDS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create missing Exaflow algorithm placeholder artifacts."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scaffold placeholders for all algorithms in the runtime catalog.",
    )
    parser.add_argument(
        "--algorithms",
        help="Comma-separated algorithm names. If set, overrides --all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned operations without writing files.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the exaflow repository root.",
    )
    return parser.parse_args()


def ensure_repo_root(path: str) -> Path:
    repo_root = Path(path).resolve()
    if not (repo_root / "pyproject.toml").exists():
        raise ValueError(f"Invalid repo root: {repo_root}")
    if not (repo_root / "exaflow" / "algorithms" / "exareme3").exists():
        raise ValueError(f"Not an exaflow repository root: {repo_root}")
    return repo_root


def parse_algorithm_list(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--algorithms provided but empty.")
    invalid = [name for name in parts if not re.fullmatch(r"[a-z0-9_]+", name)]
    if invalid:
        raise ValueError(
            f"Invalid algorithm identifiers: {', '.join(sorted(set(invalid)))}"
        )
    return sorted(set(parts))


def load_runtime_catalog(repo_root: Path) -> list[str]:
    try:
        sys.path.insert(0, str(repo_root))
        import exaflow  # pylint: disable=import-outside-toplevel

        return sorted(exaflow.exareme3_algorithm_classes.keys())
    except Exception:  # pylint: disable=broad-except
        probe = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-c",
                (
                    "import json, sys;"
                    "sys.path.insert(0, '.');"
                    "import exaflow;"
                    "print(json.dumps(sorted(exaflow.exareme3_algorithm_classes.keys())))"
                ),
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            message = probe.stderr.strip() or probe.stdout.strip() or "Unknown error"
            raise RuntimeError(f"Failed to load runtime catalog via poetry: {message}")

        output = probe.stdout.strip()
        if not output:
            raise RuntimeError("Poetry probe returned empty runtime catalog output.")
        return json.loads(output)


def snake_to_title(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split("_"))


def infer_standalone_folder(algorithm: str, repo_root: Path) -> str:
    base = repo_root / "tests" / "standalone_tests" / "federated_algorithms"

    for folder, algorithms in STANDALONE_FOLDER_RULES.items():
        if algorithm in algorithms and (base / folder).exists():
            return folder

    prefix_based = {
        "naive_bayes": "naive_bayes",
        "ttest": "statistics",
        "anova": "statistics",
    }
    for prefix, folder in prefix_based.items():
        if algorithm.startswith(prefix) and (base / folder).exists():
            return folder

    return "_generated"


def algorithm_template(algorithm: str) -> str:
    class_name = "".join(part.capitalize() for part in algorithm.split("_"))
    return f'''from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm


class {class_name}(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="{algorithm}",
            desc="TODO: describe the algorithm behavior and assumptions.",
            label="{snake_to_title(algorithm)}",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Dependent variable",
                    desc="TODO: define dependent variable constraints.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Independent variables",
                    desc="TODO: define independent variable constraints.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=True,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={{}},
            type=specs.AlgorithmType.EXAREME3,
            components=[],
        )

    def run(self):
        raise NotImplementedError("TODO: implement algorithm runtime logic.")
'''


def standalone_test_template(algorithm: str) -> str:
    test_name = algorithm.replace("-", "_")
    return f"""def test_{test_name}_standalone_placeholder():
    raise NotImplementedError(
        "TODO: replace placeholder with federated standalone tests for {algorithm}."
    )
"""


def prod_validation_test_template(algorithm: str) -> str:
    test_name = algorithm.replace("-", "_")
    return f'''from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "{algorithm}"
expected_file = Path(__file__).parent / "expected" / f"{{algorithm_name}}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_{test_name}_validation(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)
    assert result
'''


def prod_expected_template() -> str:
    return json.dumps({"test_cases": []}, indent=2) + "\n"


def docs_template(algorithm: str) -> str:
    return f"""# {snake_to_title(algorithm)}

## Overview

TODO: document algorithm purpose, inputs, outputs, and constraints.

## API Contract

- **Algorithm name**: `{algorithm}`
- **Type**: `exareme3`
- **Status**: placeholder

## Validation Notes

TODO: add standalone and production validation strategy.
"""


def to_rel_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def register(
    report: list[ReportEntry],
    *,
    algorithm: str,
    check: str,
    status: str,
    message: str,
    path: Path | None,
    repo_root: Path,
) -> None:
    report.append(
        ReportEntry(
            algorithm=algorithm,
            phase="scaffold",
            check=check,
            status=status,
            message=message,
            path=to_rel_path(path, repo_root) if path else None,
        )
    )


def write_placeholder(
    report: list[ReportEntry],
    *,
    algorithm: str,
    check: str,
    path: Path,
    content: str,
    dry_run: bool,
    repo_root: Path,
) -> None:
    if path.exists():
        register(
            report,
            algorithm=algorithm,
            check=check,
            status="skipped_existing",
            message="File already exists; no overwrite performed.",
            path=path,
            repo_root=repo_root,
        )
        return

    if dry_run:
        register(
            report,
            algorithm=algorithm,
            check=check,
            status="created",
            message="Dry-run: file would be created.",
            path=path,
            repo_root=repo_root,
        )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    register(
        report,
        algorithm=algorithm,
        check=check,
        status="created",
        message="File created.",
        path=path,
        repo_root=repo_root,
    )


def run_for_algorithm(
    algorithm: str,
    report: list[ReportEntry],
    *,
    repo_root: Path,
    dry_run: bool,
) -> None:
    standalone_folder = infer_standalone_folder(algorithm, repo_root)

    module_path = repo_root / "exaflow" / "algorithms" / "exareme3" / f"{algorithm}.py"
    standalone_test = (
        repo_root
        / "tests"
        / "standalone_tests"
        / "federated_algorithms"
        / standalone_folder
        / f"test_{algorithm}.py"
    )
    prod_test = (
        repo_root / "tests" / "prod_env_tests" / f"test_{algorithm}_validation.py"
    )
    prod_expected = (
        repo_root
        / "tests"
        / "prod_env_tests"
        / "expected"
        / f"{algorithm}_expected.json"
    )
    doc_path = repo_root / "documentation" / "algorithms" / f"{algorithm}.md"

    write_placeholder(
        report,
        algorithm=algorithm,
        check="algorithm_module",
        path=module_path,
        content=algorithm_template(algorithm),
        dry_run=dry_run,
        repo_root=repo_root,
    )
    write_placeholder(
        report,
        algorithm=algorithm,
        check="standalone_test",
        path=standalone_test,
        content=standalone_test_template(algorithm),
        dry_run=dry_run,
        repo_root=repo_root,
    )
    write_placeholder(
        report,
        algorithm=algorithm,
        check="prod_env_test",
        path=prod_test,
        content=prod_validation_test_template(algorithm),
        dry_run=dry_run,
        repo_root=repo_root,
    )
    write_placeholder(
        report,
        algorithm=algorithm,
        check="prod_env_expected",
        path=prod_expected,
        content=prod_expected_template(),
        dry_run=dry_run,
        repo_root=repo_root,
    )
    write_placeholder(
        report,
        algorithm=algorithm,
        check="documentation",
        path=doc_path,
        content=docs_template(algorithm),
        dry_run=dry_run,
        repo_root=repo_root,
    )


def summarize(report: Iterable[ReportEntry]) -> dict:
    report_dicts = [entry.to_dict() for entry in report]
    created = [entry for entry in report_dicts if entry["status"] == "created"]
    skipped_existing = [
        entry for entry in report_dicts if entry["status"] == "skipped_existing"
    ]
    failed = [entry for entry in report_dicts if entry["status"] == "failed"]
    return {
        "created": created,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "report": report_dicts,
    }


def main() -> int:
    args = parse_args()
    repo_root = ensure_repo_root(args.repo_root)

    try:
        runtime_catalog = load_runtime_catalog(repo_root)
    except Exception as exc:  # pylint: disable=broad-except
        print(
            json.dumps(
                {
                    "created": [],
                    "skipped_existing": [],
                    "failed": [
                        {
                            "algorithm": "*",
                            "phase": "scaffold",
                            "check": "load_runtime_catalog",
                            "status": "failed",
                            "message": str(exc),
                            "path": None,
                        }
                    ],
                    "report": [],
                },
                indent=2,
            )
        )
        return 1

    if args.algorithms:
        target_algorithms = parse_algorithm_list(args.algorithms)
    else:
        target_algorithms = runtime_catalog

    unknown = [name for name in target_algorithms if name not in runtime_catalog]
    if unknown:
        print(
            json.dumps(
                {
                    "created": [],
                    "skipped_existing": [],
                    "failed": [
                        {
                            "algorithm": name,
                            "phase": "scaffold",
                            "check": "target_validation",
                            "status": "failed",
                            "message": "Algorithm not found in runtime catalog.",
                            "path": None,
                        }
                        for name in unknown
                    ],
                    "report": [],
                },
                indent=2,
            )
        )
        return 1

    report: list[ReportEntry] = []
    for algorithm in target_algorithms:
        try:
            run_for_algorithm(
                algorithm,
                report,
                repo_root=repo_root,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # pylint: disable=broad-except
            register(
                report,
                algorithm=algorithm,
                check="scaffold_algorithm",
                status="failed",
                message=str(exc),
                path=None,
                repo_root=repo_root,
            )

    summary = summarize(report)
    summary["mode"] = "dry-run" if args.dry_run else "apply"
    summary["targets"] = target_algorithms
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
