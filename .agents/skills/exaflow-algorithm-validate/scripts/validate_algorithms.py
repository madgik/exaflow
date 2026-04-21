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

LEGACY_PROD_TEST_PATHS = {
    "anova_twoway": "tests/prod_env_tests/test_anova_twoway.py",
    "describe": "tests/prod_env_tests/test_describe.py",
    "histogram": "tests/prod_env_tests/test_histogram.py",
    "kmeans": "tests/prod_env_tests/test_kmeans.py",
    "linear_svm": "tests/prod_env_tests/test_linear_svm.py",
    "naive_bayes_categorical_cv": "tests/prod_env_tests/test_naive_bayes_categorical_cv.py",
    "naive_bayes_gaussian_cv": "tests/prod_env_tests/test_naive_bayes_gaussian_cv.py",
    "ttest_independent": "tests/prod_env_tests/test_independent_ttest.py",
    "ttest_onesample": "tests/prod_env_tests/test_one_sample.py",
    "ttest_paired": "tests/prod_env_tests/test_paired_ttest.py",
}

LEGACY_EXPECTED_PATHS = {
    "linear_svm": "tests/prod_env_tests/expected/svm_scikit_expected.json",
    "naive_bayes_gaussian_cv": "tests/prod_env_tests/expected/naive_bayes_gauss_cv_expected.json",
}

LEGACY_DOC_PATHS = {
    "anova_oneway": "documentation/algorithms/ANOVA.md",
    "anova_twoway": "documentation/algorithms/ANOVA.md",
    "describe": "documentation/algorithms/Describe.md",
    "histogram": "documentation/algorithms/Histogram.md",
    "kmeans": "documentation/algorithms/k-means.md",
    "linear_regression": "documentation/algorithms/LinearRegression.md",
    "linear_regression_cv": "documentation/algorithms/LinearRegression.md",
    "logistic_regression": "documentation/algorithms/LogisticRegression.md",
    "logistic_regression_cv": "documentation/algorithms/LogisticRegression.md",
    "naive_bayes_categorical": "documentation/algorithms/NaiveBayes.md",
    "naive_bayes_categorical_cv": "documentation/algorithms/NaiveBayes.md",
    "naive_bayes_gaussian": "documentation/algorithms/NaiveBayes.md",
    "naive_bayes_gaussian_cv": "documentation/algorithms/NaiveBayes.md",
    "pca": "documentation/algorithms/PCA.md",
    "pca_with_transformation": "documentation/algorithms/PCA.md",
    "pearson_correlation": "documentation/algorithms/Pearson.md",
    "linear_svm": "documentation/algorithms/SVM.md",
    "ttest_independent": "documentation/algorithms/TtestIndependent.md",
    "ttest_onesample": "documentation/algorithms/TtestOneSample.md",
    "ttest_paired": "documentation/algorithms/TtestPaired.md",
}

LEGACY_STANDALONE_PATHS = {
    "linear_regression": "tests/standalone_tests/federated_algorithms/linear_model/test_ols.py",
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
        description="Validate required algorithm development artifacts and checks."
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Validate only changed algorithms. Enabled by default when --algorithms is not set.",
    )
    parser.add_argument(
        "--algorithms",
        help="Comma-separated algorithm names. Overrides --changed-only.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run strict tier: standalone + prod_env tests.",
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


def to_rel(path: Path | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def register(
    report: list[ReportEntry],
    *,
    algorithm: str,
    phase: str,
    check: str,
    status: str,
    message: str,
    path: Path | None,
    repo_root: Path,
) -> None:
    report.append(
        ReportEntry(
            algorithm=algorithm,
            phase=phase,
            check=check,
            status=status,
            message=message,
            path=to_rel(path, repo_root),
        )
    )


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


def get_changed_files(repo_root: Path) -> list[str]:
    diff_cmd = [
        "git",
        "-C",
        str(repo_root),
        "diff",
        "--name-only",
        "--relative",
        "HEAD",
    ]
    untracked_cmd = [
        "git",
        "-C",
        str(repo_root),
        "ls-files",
        "--others",
        "--exclude-standard",
    ]

    diff_result = subprocess.run(diff_cmd, capture_output=True, text=True, check=False)
    untracked_result = subprocess.run(
        untracked_cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    files = set()
    if diff_result.returncode == 0:
        files.update(line.strip() for line in diff_result.stdout.splitlines() if line)
    if untracked_result.returncode == 0:
        files.update(
            line.strip() for line in untracked_result.stdout.splitlines() if line
        )
    return sorted(files)


def _legacy_reverse_map(mapping: dict[str, str]) -> dict[str, str]:
    return {value: key for key, value in mapping.items()}


def map_changed_files_to_algorithms(
    changed_files: Iterable[str],
    runtime_catalog: set[str],
) -> set[str]:
    algorithms: set[str] = set()

    reverse_prod = _legacy_reverse_map(LEGACY_PROD_TEST_PATHS)
    reverse_docs = _legacy_reverse_map(LEGACY_DOC_PATHS)
    reverse_standalone = _legacy_reverse_map(LEGACY_STANDALONE_PATHS)

    patterns = [
        re.compile(r"^exaflow/algorithms/exareme3/([a-z0-9_]+)\\.py$"),
        re.compile(r"^tests/prod_env_tests/test_([a-z0-9_]+)_validation\\.py$"),
        re.compile(r"^tests/prod_env_tests/expected/([a-z0-9_]+)_expected\\.json$"),
        re.compile(
            r"^tests/standalone_tests/federated_algorithms/.*/test_([a-z0-9_]+)\\.py$"
        ),
        re.compile(r"^documentation/algorithms/([a-z0-9_]+)\\.md$"),
    ]

    for changed in changed_files:
        if changed in reverse_prod:
            algorithms.add(reverse_prod[changed])
            continue
        if changed in reverse_docs:
            algorithms.add(reverse_docs[changed])
            continue
        if changed in reverse_standalone:
            algorithms.add(reverse_standalone[changed])
            continue

        for pattern in patterns:
            match = pattern.match(changed)
            if not match:
                continue
            candidate = match.group(1)
            if candidate in runtime_catalog:
                algorithms.add(candidate)
            break

    return algorithms


def resolve_path_with_legacy(
    preferred: Path,
    legacy_candidate: Path | None,
) -> tuple[bool, Path | None, str]:
    if preferred.exists():
        return True, preferred, "Found preferred path."
    if legacy_candidate and legacy_candidate.exists():
        return True, legacy_candidate, "Found legacy compatibility path."
    if legacy_candidate:
        return (
            False,
            preferred,
            f"Missing preferred path and compatibility path ({legacy_candidate}).",
        )
    return False, preferred, "Required path not found."


def discover_standalone_paths(repo_root: Path, algorithm: str) -> list[Path]:
    base = repo_root / "tests" / "standalone_tests" / "federated_algorithms"
    matches = sorted(base.glob(f"**/test_{algorithm}.py"))
    if matches:
        return matches

    legacy = LEGACY_STANDALONE_PATHS.get(algorithm)
    if legacy:
        legacy_path = repo_root / legacy
        if legacy_path.exists():
            return [legacy_path]

    return []


def check_import_and_spec(
    algorithm: str,
    report: list[ReportEntry],
    *,
    repo_root: Path,
) -> None:
    module_path = repo_root / "exaflow" / "algorithms" / "exareme3" / f"{algorithm}.py"
    if not module_path.exists():
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="algorithm_module_exists",
            status="failed",
            message="Algorithm module file not found.",
            path=module_path,
            repo_root=repo_root,
        )
        return

    register(
        report,
        algorithm=algorithm,
        phase="static",
        check="algorithm_module_exists",
        status="pass",
        message="Algorithm module exists.",
        path=module_path,
        repo_root=repo_root,
    )

    probe = subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "-c",
            (
                "import importlib, inspect, json, sys;"
                "sys.path.insert(0, '.');"
                f"module = importlib.import_module('exaflow.algorithms.exareme3.{algorithm}');"
                "from exaflow.algorithms.exareme3.utils.algorithm import Algorithm;"
                "classes = ["
                "cls for _, cls in inspect.getmembers(module, inspect.isclass) "
                "if cls.__module__ == module.__name__ and issubclass(cls, Algorithm) and cls is not Algorithm"
                "];"
                f"match = any(cls.get_specification().name == '{algorithm}' for cls in classes);"
                "print(json.dumps({'classes_found': len(classes), 'match': match}))"
            ),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    if probe.returncode != 0:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="algorithm_module_import",
            status="failed",
            message=probe.stderr.strip()
            or probe.stdout.strip()
            or "Module import probe failed.",
            path=module_path,
            repo_root=repo_root,
        )
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="algorithm_spec_name_match",
            status="failed",
            message="Spec probe skipped because module import probe failed.",
            path=module_path,
            repo_root=repo_root,
        )
        return

    register(
        report,
        algorithm=algorithm,
        phase="static",
        check="algorithm_module_import",
        status="pass",
        message="Module import succeeded.",
        path=module_path,
        repo_root=repo_root,
    )

    try:
        payload = json.loads(probe.stdout.strip())
        classes_found = int(payload.get("classes_found", 0))
        match = bool(payload.get("match", False))
    except Exception:  # pylint: disable=broad-except
        classes_found = 0
        match = False

    if classes_found == 0:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="algorithm_spec_name_match",
            status="failed",
            message="No Algorithm subclass found in module.",
            path=module_path,
            repo_root=repo_root,
        )
    elif match:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="algorithm_spec_name_match",
            status="pass",
            message="get_specification().name matches algorithm identifier.",
            path=module_path,
            repo_root=repo_root,
        )
    else:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="algorithm_spec_name_match",
            status="failed",
            message="No Algorithm subclass has get_specification().name matching the algorithm.",
            path=module_path,
            repo_root=repo_root,
        )


def check_required_paths(
    algorithm: str,
    report: list[ReportEntry],
    *,
    repo_root: Path,
) -> None:
    standalone_paths = discover_standalone_paths(repo_root, algorithm)
    if standalone_paths:
        primary = standalone_paths[0]
        message = "Found standalone test path."
        if (
            algorithm in LEGACY_STANDALONE_PATHS
            and to_rel(primary, repo_root) == LEGACY_STANDALONE_PATHS[algorithm]
        ):
            message = "Found standalone test via legacy compatibility path."
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="standalone_test_exists",
            status="pass",
            message=message,
            path=primary,
            repo_root=repo_root,
        )
    else:
        expected = (
            repo_root
            / "tests"
            / "standalone_tests"
            / "federated_algorithms"
            / "_generated"
            / f"test_{algorithm}.py"
        )
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="standalone_test_exists",
            status="failed",
            message="No standalone test found under federated_algorithms.",
            path=expected,
            repo_root=repo_root,
        )

    preferred_prod = (
        repo_root / "tests" / "prod_env_tests" / f"test_{algorithm}_validation.py"
    )
    legacy_prod = (
        repo_root / LEGACY_PROD_TEST_PATHS[algorithm]
        if algorithm in LEGACY_PROD_TEST_PATHS
        else None
    )
    ok, resolved, msg = resolve_path_with_legacy(preferred_prod, legacy_prod)
    register(
        report,
        algorithm=algorithm,
        phase="static",
        check="prod_env_test_exists",
        status="pass" if ok else "failed",
        message=msg,
        path=resolved,
        repo_root=repo_root,
    )

    preferred_expected = (
        repo_root
        / "tests"
        / "prod_env_tests"
        / "expected"
        / f"{algorithm}_expected.json"
    )
    legacy_expected = (
        repo_root / LEGACY_EXPECTED_PATHS[algorithm]
        if algorithm in LEGACY_EXPECTED_PATHS
        else None
    )
    ok, resolved, msg = resolve_path_with_legacy(preferred_expected, legacy_expected)
    register(
        report,
        algorithm=algorithm,
        phase="static",
        check="prod_env_expected_exists",
        status="pass" if ok else "failed",
        message=msg,
        path=resolved,
        repo_root=repo_root,
    )

    preferred_doc = repo_root / "documentation" / "algorithms" / f"{algorithm}.md"
    legacy_doc = (
        repo_root / LEGACY_DOC_PATHS[algorithm]
        if algorithm in LEGACY_DOC_PATHS
        else None
    )
    ok, resolved, msg = resolve_path_with_legacy(preferred_doc, legacy_doc)
    register(
        report,
        algorithm=algorithm,
        phase="static",
        check="documentation_exists",
        status="pass" if ok else "failed",
        message=msg,
        path=resolved,
        repo_root=repo_root,
    )


def run_command(
    command: list[str],
    *,
    cwd: Path,
) -> tuple[int, str, str]:
    proc = subprocess.run(
        command, cwd=str(cwd), capture_output=True, text=True, check=False
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _collect_candidate_lint_files(
    repo_root: Path,
    changed_files: list[str],
    algorithms: list[str],
) -> list[str]:
    lint_files = [
        f for f in changed_files if f.endswith(".py") and (repo_root / f).exists()
    ]

    if lint_files:
        return sorted(set(lint_files))

    generated = []
    for algorithm in algorithms:
        module = f"exaflow/algorithms/exareme3/{algorithm}.py"
        if (repo_root / module).exists():
            generated.append(module)

        standalone_paths = discover_standalone_paths(repo_root, algorithm)
        generated.extend(to_rel(path, repo_root) for path in standalone_paths if path)

        prod_test = f"tests/prod_env_tests/test_{algorithm}_validation.py"
        legacy_prod = LEGACY_PROD_TEST_PATHS.get(algorithm)
        for candidate in [prod_test, legacy_prod]:
            if candidate and (repo_root / candidate).exists():
                generated.append(candidate)

    return sorted(set(path for path in generated if path))


def run_fast_tier(
    report: list[ReportEntry],
    *,
    algorithms: list[str],
    repo_root: Path,
    changed_files: list[str],
) -> None:
    lint_files = _collect_candidate_lint_files(repo_root, changed_files, algorithms)

    if lint_files:
        rc, _, stderr = run_command(
            ["poetry", "run", "ruff", "check", "--select", "I", *lint_files],
            cwd=repo_root,
        )
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="ruff_check_select_I",
            status="pass" if rc == 0 else "failed",
            message="ruff check --select I passed."
            if rc == 0
            else (stderr or "ruff check failed."),
            path=None,
            repo_root=repo_root,
        )

        rc, _, stderr = run_command(
            ["poetry", "run", "ruff", "format", "--check", *lint_files],
            cwd=repo_root,
        )
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="ruff_format_check",
            status="pass" if rc == 0 else "failed",
            message="ruff format --check passed."
            if rc == 0
            else (stderr or "ruff format check failed."),
            path=None,
            repo_root=repo_root,
        )
    else:
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="ruff_checks",
            status="pass",
            message="No Python files selected for lint checks.",
            path=None,
            repo_root=repo_root,
        )

    standalone_files: list[str] = []
    for algorithm in algorithms:
        for path in discover_standalone_paths(repo_root, algorithm):
            standalone_files.append(to_rel(path, repo_root) or "")
    standalone_files = sorted(set(path for path in standalone_files if path))

    if standalone_files:
        rc, _, stderr = run_command(
            ["poetry", "run", "pytest", "--verbosity=2", *standalone_files],
            cwd=repo_root,
        )
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="standalone_tests",
            status="pass" if rc == 0 else "failed",
            message="Standalone tests passed."
            if rc == 0
            else (stderr or "Standalone tests failed."),
            path=None,
            repo_root=repo_root,
        )
    else:
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="standalone_tests",
            status="failed",
            message="No standalone tests found for selected algorithms.",
            path=None,
            repo_root=repo_root,
        )


def resolve_prod_test_file(repo_root: Path, algorithm: str) -> str | None:
    preferred = f"tests/prod_env_tests/test_{algorithm}_validation.py"
    if (repo_root / preferred).exists():
        return preferred

    legacy = LEGACY_PROD_TEST_PATHS.get(algorithm)
    if legacy and (repo_root / legacy).exists():
        return legacy

    return None


def run_strict_tier(
    report: list[ReportEntry],
    *,
    algorithms: list[str],
    repo_root: Path,
) -> None:
    prod_files = sorted(
        {
            path
            for algorithm in algorithms
            for path in [resolve_prod_test_file(repo_root, algorithm)]
            if path
        }
    )

    if prod_files:
        rc, _, stderr = run_command(
            ["poetry", "run", "pytest", "--verbosity=2", *prod_files],
            cwd=repo_root,
        )
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="prod_env_tests",
            status="pass" if rc == 0 else "failed",
            message="prod_env tests passed."
            if rc == 0
            else (stderr or "prod_env tests failed."),
            path=None,
            repo_root=repo_root,
        )
    else:
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="prod_env_tests",
            status="failed",
            message="No prod_env tests found for selected algorithms.",
            path=None,
            repo_root=repo_root,
        )


def summarize(report: list[ReportEntry]) -> dict:
    rows = [entry.to_dict() for entry in report]
    failed = [row for row in rows if row["status"] == "failed"]
    passed = [row for row in rows if row["status"] == "pass"]
    return {
        "passed": passed,
        "failed": failed,
        "report": rows,
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
                    "passed": [],
                    "failed": [
                        {
                            "algorithm": "*",
                            "phase": "static",
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

    changed_files = get_changed_files(repo_root)

    if args.algorithms:
        target_algorithms = parse_algorithm_list(args.algorithms)
    else:
        use_changed_only = True
        if args.changed_only:
            use_changed_only = True

        if use_changed_only:
            target_algorithms = sorted(
                map_changed_files_to_algorithms(changed_files, set(runtime_catalog))
            )
        else:
            target_algorithms = list(runtime_catalog)

    unknown = [name for name in target_algorithms if name not in runtime_catalog]
    if unknown:
        print(
            json.dumps(
                {
                    "passed": [],
                    "failed": [
                        {
                            "algorithm": name,
                            "phase": "static",
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

    if not target_algorithms:
        register(
            report,
            algorithm="*",
            phase="static",
            check="target_selection",
            status="pass",
            message="No changed algorithms detected. Nothing to validate.",
            path=None,
            repo_root=repo_root,
        )
        summary = summarize(report)
        summary["targets"] = target_algorithms
        summary["tier"] = "strict" if args.strict else "fast"
        print(json.dumps(summary, indent=2))
        return 0

    for algorithm in target_algorithms:
        check_import_and_spec(algorithm, report, repo_root=repo_root)
        check_required_paths(algorithm, report, repo_root=repo_root)

    run_fast_tier(
        report,
        algorithms=target_algorithms,
        repo_root=repo_root,
        changed_files=changed_files,
    )

    if args.strict:
        run_strict_tier(
            report,
            algorithms=target_algorithms,
            repo_root=repo_root,
        )

    summary = summarize(report)
    summary["targets"] = target_algorithms
    summary["tier"] = "strict" if args.strict else "fast"
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
