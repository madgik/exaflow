#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPORT_FIELDS = (
    "algorithm",
    "phase",
    "check",
    "status",
    "severity",
    "message",
    "next_action",
    "path",
)

ALGORITHM_ID_RE = re.compile(r"[a-z][a-z0-9_]*")
PLACEHOLDER_PATTERN = re.compile(
    r"\bTODO\b|NotImplementedError|__REPLACE_ME_[A-Z0-9_]*__"
)

PREFERRED_DOC_PATHS = {
    "anova_oneway": "documentation/algorithms/ANOVAOneWay.md",
    "anova_twoway": "documentation/algorithms/ANOVATwoWay.md",
    "chi_squared": "documentation/algorithms/ChiSquared.md",
    "describe": "documentation/algorithms/Describe.md",
    "fisher_exact": "documentation/algorithms/FisherExact.md",
    "histogram": "documentation/algorithms/Histogram.md",
    "kmeans": "documentation/algorithms/k-means.md",
    "linear_regression": "documentation/algorithms/LinearRegression.md",
    "linear_regression_cv": "documentation/algorithms/LinearRegression.md",
    "logistic_regression": "documentation/algorithms/LogisticRegression.md",
    "logistic_regression_cv": "documentation/algorithms/LogisticRegression.md",
    "naive_bayes_categorical": "documentation/algorithms/NaiveBayesCategorical.md",
    "naive_bayes_categorical_cv": "documentation/algorithms/NaiveBayesCategoricalCV.md",
    "naive_bayes_gaussian": "documentation/algorithms/NaiveBayesGaussian.md",
    "naive_bayes_gaussian_cv": "documentation/algorithms/NaiveBayesGaussianCV.md",
    "pca": "documentation/algorithms/PCA.md",
    "pca_with_transformation": "documentation/algorithms/PCAWithTransformation.md",
    "pearson_correlation": "documentation/algorithms/Pearson.md",
    "linear_svm": "documentation/algorithms/SVM.md",
    "ttest_independent": "documentation/algorithms/TtestIndependent.md",
    "ttest_onesample": "documentation/algorithms/TtestOneSample.md",
    "ttest_paired": "documentation/algorithms/TtestPaired.md",
}

ALGORITHM_PROFILE_SYSTEM_PREFIXES = (
    "configs/",
    "exadata-validator/",
    "exaflow/aggregation_server/",
    "exaflow/controller/",
    "exaflow/protos/",
    "exaflow/worker/",
    "kubernetes/",
)
ALGORITHM_PROFILE_SYSTEM_FILES = {
    ".deployment.sample.toml",
    "pyproject.toml",
    "run_analysis",
    "tasks.py",
    "uv.lock",
}
ALGORITHM_PROFILE_SHARED_FILES = {
    "exaflow/algorithms/federated/README.md",
    "exaflow/algorithms/federated/__init__.py",
    "exaflow/algorithms/specifications.py",
}
ALGORITHM_PROFILE_SHARED_PATTERNS = (
    re.compile(r"^exaflow/algorithms/federated/[a-z0-9_]+/__init__\.py$"),
)
ALGORITHM_PROFILE_ALLOWED_PATTERNS = (
    re.compile(r"^documentation/algorithms/.+\.md$"),
    re.compile(r"^exaflow/algorithms/exareme3/[a-z][a-z0-9_]*\.py$"),
    re.compile(r"^exaflow/algorithms/federated/.+"),
    re.compile(r"^tests/prod_env_tests/test_[a-z0-9_]+\.py$"),
    re.compile(r"^tests/prod_env_tests/expected/[a-z0-9_]+_expected\.json$"),
    re.compile(r"^tests/standalone_tests/federated_algorithms/.+"),
)


@dataclass
class ReportEntry:
    algorithm: str
    phase: str
    check: str
    status: str
    severity: str
    message: str
    next_action: str | None
    path: str | None

    def to_dict(self) -> dict:
        return {field: getattr(self, field) for field in REPORT_FIELDS}


@dataclass
class AlgorithmPaths:
    standalone: Path | None = None
    prod_test: Path | None = None
    prod_expected: Path | None = None
    documentation: Path | None = None
    standalone_canonical: bool = False
    prod_test_canonical: bool = False
    prod_expected_canonical: bool = False
    documentation_canonical: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Exaflow algorithm development artifacts and checks."
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Validate only algorithms detected from changed files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Validate all runtime-catalog algorithms. When omitted, changed-only "
            "selection is used unless --algorithms/--new-algorithm is provided."
        ),
    )
    parser.add_argument(
        "--algorithms",
        help="Comma-separated algorithm names. Overrides --changed-only/--all.",
    )
    parser.add_argument(
        "--new-algorithm",
        help=(
            "Comma-separated newly added algorithms. Enables full canonical "
            "integration checks."
        ),
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
        raise ValueError("Algorithm list provided but empty.")
    invalid = [name for name in parts if not ALGORITHM_ID_RE.fullmatch(name)]
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
    severity: str,
    message: str,
    path: Path | None,
    repo_root: Path,
    next_action: str | None = None,
) -> None:
    report.append(
        ReportEntry(
            algorithm=algorithm,
            phase=phase,
            check=check,
            status=status,
            severity=severity,
            message=message,
            next_action=next_action,
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
                "uv",
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
            raise RuntimeError(f"Failed to load runtime catalog via uv: {message}")

        output = probe.stdout.strip()
        if not output:
            raise RuntimeError("uv probe returned empty runtime catalog output.")
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


def _reverse_path_map(mapping: dict[str, str]) -> dict[str, str]:
    return {value: key for key, value in mapping.items()}


def map_changed_files_to_algorithms(changed_files: Iterable[str]) -> set[str]:
    algorithms: set[str] = set()

    reverse_docs = _reverse_path_map(PREFERRED_DOC_PATHS)

    patterns = [
        re.compile(r"^exaflow/algorithms/exareme3/([a-z][a-z0-9_]*)\.py$"),
        re.compile(r"^exaflow/algorithms/federated/[a-z0-9_]+/([a-z][a-z0-9_]*)\.py$"),
        re.compile(r"^tests/prod_env_tests/test_([a-z0-9_]+)\.py$"),
        re.compile(r"^tests/prod_env_tests/expected/([a-z0-9_]+)_expected\.json$"),
        re.compile(
            r"^tests/standalone_tests/federated_algorithms/.*/test_([a-z0-9_]+)\.py$"
        ),
        re.compile(r"^documentation/algorithms/([a-z0-9_]+)\.md$"),
    ]

    for changed in changed_files:
        if changed in reverse_docs:
            algorithms.add(reverse_docs[changed])
            continue

        for pattern in patterns:
            match = pattern.match(changed)
            if not match:
                continue
            algorithms.add(match.group(1))
            break

    return algorithms


def is_algorithm_profile_shared_path(changed_file: str) -> bool:
    if changed_file in ALGORITHM_PROFILE_SHARED_FILES:
        return True
    return any(
        pattern.match(changed_file)
        for pattern in ALGORITHM_PROFILE_SHARED_PATTERNS
    )


def is_algorithm_profile_allowed_path(changed_file: str) -> bool:
    if changed_file in ALGORITHM_PROFILE_SYSTEM_FILES:
        return False
    if changed_file.startswith(ALGORITHM_PROFILE_SYSTEM_PREFIXES):
        return False
    if changed_file.startswith("exaflow/algorithms/federated/docs/"):
        return True
    if is_algorithm_profile_shared_path(changed_file):
        return True
    return any(
        pattern.match(changed_file)
        for pattern in ALGORITHM_PROFILE_ALLOWED_PATTERNS
    )


def algorithm_profile_boundary_violations(changed_files: Iterable[str]) -> list[str]:
    return sorted(
        changed_file
        for changed_file in changed_files
        if not is_algorithm_profile_allowed_path(changed_file)
    )


def check_algorithm_profile_boundary(
    report: list[ReportEntry],
    *,
    repo_root: Path,
    changed_files: list[str],
) -> None:
    violations = algorithm_profile_boundary_violations(changed_files)

    if not violations:
        register(
            report,
            algorithm="*",
            phase="static",
            check="algorithm_profile_boundary",
            status="pass",
            severity="pass",
            message="Changed files stay inside the algorithm developer profile.",
            path=None,
            repo_root=repo_root,
        )
        return

    for rel in violations:
        register(
            report,
            algorithm="*",
            phase="static",
            check="algorithm_profile_boundary",
            status="failed",
            severity="failed",
            message=(
                "Algorithm profile cannot change system-owned files: " f"{rel}"
            ),
            path=repo_root / rel,
            repo_root=repo_root,
            next_action=(
                "Stop algorithm implementation and write a System Feature Request "
                "covering the needed capability, current limitation, minimal "
                "system interface, algorithm impact, and evidence."
            ),
        )


def select_target_algorithms(
    *,
    args: argparse.Namespace,
    runtime_catalog: list[str],
    changed_files: list[str],
) -> tuple[list[str], set[str], str]:
    if args.new_algorithm and args.algorithms:
        raise ValueError("Use either --new-algorithm or --algorithms, not both.")
    if args.changed_only and args.all:
        raise ValueError("Use either --changed-only or --all, not both.")

    if args.new_algorithm:
        targets = parse_algorithm_list(args.new_algorithm)
        return targets, set(targets), "new-algorithm"

    if args.algorithms:
        targets = parse_algorithm_list(args.algorithms)
        return targets, set(), "explicit"

    use_changed_only = args.changed_only or not args.all
    if use_changed_only:
        targets = sorted(map_changed_files_to_algorithms(changed_files))
        return targets, set(), "changed-only"

    return sorted(set(runtime_catalog)), set(), "all"


def discover_canonical_standalone_paths(repo_root: Path, algorithm: str) -> list[Path]:
    base = repo_root / "tests" / "standalone_tests" / "federated_algorithms"
    return sorted(base.glob(f"**/test_{algorithm}.py"))


def discover_standalone_paths(repo_root: Path, algorithm: str) -> list[Path]:
    return discover_canonical_standalone_paths(repo_root, algorithm)


def check_import_and_spec(
    algorithm: str,
    report: list[ReportEntry],
    *,
    repo_root: Path,
    runtime_catalog_set: set[str],
) -> None:
    module_path = repo_root / "exaflow" / "algorithms" / "exareme3" / f"{algorithm}.py"

    if algorithm in runtime_catalog_set:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="runtime_catalog_membership",
            status="pass",
            severity="pass",
            message="Algorithm present in runtime catalog.",
            path=None,
            repo_root=repo_root,
        )
    else:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="runtime_catalog_membership",
            status="failed",
            severity="failed",
            message="Algorithm is not present in runtime catalog.",
            path=None,
            repo_root=repo_root,
            next_action=(
                "Ensure exaflow.algorithms.exareme3 module/class discovery registers "
                f"'{algorithm}' in exaflow.exareme3_algorithm_classes."
            ),
        )

    if not module_path.exists():
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="algorithm_module_exists",
            status="failed",
            severity="failed",
            message="Algorithm module file not found.",
            path=module_path,
            repo_root=repo_root,
            next_action=(
                "Create the module or run scaffold: "
                "uv run python "
                ".agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py "
                f"--repo-root . --algorithms {algorithm}"
            ),
        )
        return

    register(
        report,
        algorithm=algorithm,
        phase="static",
        check="algorithm_module_exists",
        status="pass",
        severity="pass",
        message="Algorithm module exists.",
        path=module_path,
        repo_root=repo_root,
    )

    probe = subprocess.run(
        [
            "uv",
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
                "if cls.__module__ == module.__name__ and issubclass(cls, Algorithm) "
                "and cls is not Algorithm"
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
            severity="failed",
            message=probe.stderr.strip()
            or probe.stdout.strip()
            or "Module import probe failed.",
            path=module_path,
            repo_root=repo_root,
            next_action="Fix module imports and class definitions, then re-run validator.",
        )
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="algorithm_spec_name_match",
            status="failed",
            severity="failed",
            message="Spec probe skipped because module import probe failed.",
            path=module_path,
            repo_root=repo_root,
            next_action=(
                "Ensure Algorithm subclass get_specification().name equals "
                f"'{algorithm}'."
            ),
        )
        return

    register(
        report,
        algorithm=algorithm,
        phase="static",
        check="algorithm_module_import",
        status="pass",
        severity="pass",
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
            severity="failed",
            message="No Algorithm subclass found in module.",
            path=module_path,
            repo_root=repo_root,
            next_action=(
                "Define an Algorithm subclass with get_specification() and run()."
            ),
        )
    elif match:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="algorithm_spec_name_match",
            status="pass",
            severity="pass",
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
            severity="failed",
            message=(
                "No Algorithm subclass has get_specification().name matching "
                "the algorithm."
            ),
            path=module_path,
            repo_root=repo_root,
            next_action=(
                "Set get_specification().name to the exact algorithm identifier."
            ),
        )


def _resolve_required_path(
    *,
    algorithm: str,
    report: list[ReportEntry],
    repo_root: Path,
    phase: str,
    check: str,
    preferred: Path,
    canonical_fix: str,
) -> tuple[Path | None, bool]:
    if preferred.exists():
        register(
            report,
            algorithm=algorithm,
            phase=phase,
            check=check,
            status="pass",
            severity="pass",
            message="Found preferred path.",
            path=preferred,
            repo_root=repo_root,
        )
        return preferred, True

    register(
        report,
        algorithm=algorithm,
        phase=phase,
        check=check,
        status="failed",
        severity="failed",
        message="Required path not found.",
        path=preferred,
        repo_root=repo_root,
        next_action=canonical_fix,
    )
    return None, False


def check_required_paths(
    algorithm: str,
    report: list[ReportEntry],
    *,
    repo_root: Path,
    enforce_canonical: bool,
) -> AlgorithmPaths:
    paths = AlgorithmPaths()

    canonical_standalone = discover_canonical_standalone_paths(repo_root, algorithm)
    if canonical_standalone:
        primary = canonical_standalone[0]
        paths.standalone = primary
        paths.standalone_canonical = True
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="standalone_test_exists",
            status="pass",
            severity="pass",
            message="Found standalone test path.",
            path=primary,
            repo_root=repo_root,
        )
    else:
        paths.standalone = None
        paths.standalone_canonical = False
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="standalone_test_exists",
            status="failed",
            severity="failed",
            message="No standalone test found under federated_algorithms.",
            path=(
                repo_root
                / "tests"
                / "standalone_tests"
                / "federated_algorithms"
                / "_generated"
                / f"test_{algorithm}.py"
            ),
            repo_root=repo_root,
            next_action=(
                "Create standalone test file or run scaffold with --family/--subfolder."
            ),
        )

    preferred_prod = repo_root / "tests" / "prod_env_tests" / f"test_{algorithm}.py"
    paths.prod_test, paths.prod_test_canonical = _resolve_required_path(
        algorithm=algorithm,
        report=report,
        repo_root=repo_root,
        phase="static",
        check="prod_env_test_exists",
        preferred=preferred_prod,
        canonical_fix=(
            f"Create canonical prod test: tests/prod_env_tests/test_{algorithm}.py"
        ),
    )

    preferred_expected = (
        repo_root
        / "tests"
        / "prod_env_tests"
        / "expected"
        / f"{algorithm}_expected.json"
    )
    paths.prod_expected, paths.prod_expected_canonical = _resolve_required_path(
        algorithm=algorithm,
        report=report,
        repo_root=repo_root,
        phase="static",
        check="prod_env_expected_exists",
        preferred=preferred_expected,
        canonical_fix=(
            "Create canonical expected fixture: "
            f"tests/prod_env_tests/expected/{algorithm}_expected.json"
        ),
    )

    preferred_doc = repo_root / PREFERRED_DOC_PATHS.get(
        algorithm, f"documentation/algorithms/{algorithm}.md"
    )
    paths.documentation, paths.documentation_canonical = _resolve_required_path(
        algorithm=algorithm,
        report=report,
        repo_root=repo_root,
        phase="static",
        check="documentation_exists",
        preferred=preferred_doc,
        canonical_fix=(
            f"Create canonical docs file: {to_rel(preferred_doc, repo_root)}"
        ),
    )

    return paths


def find_placeholder_tokens(text: str) -> list[str]:
    found = {match.group(0) for match in PLACEHOLDER_PATTERN.finditer(text)}
    return sorted(found)


def check_placeholder_file(
    algorithm: str,
    report: list[ReportEntry],
    *,
    repo_root: Path,
    label: str,
    path: Path | None,
) -> None:
    if path is None or not path.exists():
        return

    tokens = find_placeholder_tokens(path.read_text(encoding="utf-8"))
    if tokens:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check=f"{label}_placeholder_check",
            status="failed",
            severity="failed",
            message=f"Placeholder tokens found: {', '.join(tokens)}.",
            path=path,
            repo_root=repo_root,
            next_action=(
                "Replace placeholders with concrete implementation, test, docs, "
                "or fixture values."
            ),
        )
    else:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check=f"{label}_placeholder_check",
            status="pass",
            severity="pass",
            message="No placeholder tokens found.",
            path=path,
            repo_root=repo_root,
        )


def find_federated_core_paths(repo_root: Path, algorithm: str) -> list[Path]:
    pattern = repo_root / "exaflow" / "algorithms" / "federated"
    return sorted(pattern.glob(f"*/{algorithm}.py"))


def federated_symbol_for_algorithm(algorithm: str) -> str:
    return "Federated" + "".join(part.capitalize() for part in algorithm.split("_"))


def check_fixture_content(
    algorithm: str,
    report: list[ReportEntry],
    *,
    repo_root: Path,
    fixture_path: Path | None,
    require_non_empty: bool,
) -> None:
    if fixture_path is None or not fixture_path.exists():
        return

    fixture_text = fixture_path.read_text(encoding="utf-8")
    tokens = find_placeholder_tokens(fixture_text)
    if tokens:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="prod_env_expected_placeholder_check",
            status="failed",
            severity="failed",
            message=f"Placeholder tokens found: {', '.join(tokens)}.",
            path=fixture_path,
            repo_root=repo_root,
            next_action=(
                "Replace fixture placeholders with concrete dataset, variable, "
                "and expected-output values."
            ),
        )

    try:
        payload = json.loads(fixture_text)
    except json.JSONDecodeError as exc:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="prod_env_expected_json_valid",
            status="failed",
            severity="failed",
            message=f"Invalid JSON fixture: {exc}",
            path=fixture_path,
            repo_root=repo_root,
            next_action="Fix JSON syntax in expected fixture file.",
        )
        return

    cases = payload.get("test_cases")
    if not isinstance(cases, list):
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="prod_env_expected_structure",
            status="failed",
            severity="failed",
            message="Fixture must define a list at key 'test_cases'.",
            path=fixture_path,
            repo_root=repo_root,
            next_action="Set fixture format to {'test_cases': [...]}.",
        )
        return

    if not cases:
        severity = "failed" if require_non_empty else "warn"
        status = "failed" if require_non_empty else "warn"
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="prod_env_expected_non_empty",
            status=status,
            severity=severity,
            message="Fixture test_cases is empty.",
            path=fixture_path,
            repo_root=repo_root,
            next_action=(
                "Add at least one runnable test case template with input/output fields."
            ),
        )
        return

    first = cases[0]
    if not isinstance(first, dict) or "input" not in first or "output" not in first:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="prod_env_expected_case_shape",
            status="failed",
            severity="failed",
            message="First test case must include 'input' and 'output' keys.",
            path=fixture_path,
            repo_root=repo_root,
            next_action=(
                "Use scaffold sample-fixture structure for first test case shape."
            ),
        )
        return

    register(
        report,
        algorithm=algorithm,
        phase="static",
        check="prod_env_expected_case_shape",
        status="pass",
        severity="pass",
        message="Expected fixture contains a runnable test-case skeleton.",
        path=fixture_path,
        repo_root=repo_root,
    )


def _check_token_in_file(
    report: list[ReportEntry],
    *,
    algorithm: str,
    repo_root: Path,
    path: Path,
    check: str,
    token: str,
    next_action: str,
) -> None:
    if not path.exists():
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check=check,
            status="failed",
            severity="failed",
            message="File not found.",
            path=path,
            repo_root=repo_root,
            next_action=next_action,
        )
        return

    content = path.read_text(encoding="utf-8")
    if token in content:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check=check,
            status="pass",
            severity="pass",
            message=f"Found expected token: {token}",
            path=path,
            repo_root=repo_root,
        )
    else:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check=check,
            status="failed",
            severity="failed",
            message=f"Missing expected token: {token}",
            path=path,
            repo_root=repo_root,
            next_action=next_action,
        )


def check_new_algorithm_integration(
    algorithm: str,
    report: list[ReportEntry],
    *,
    repo_root: Path,
) -> None:
    core_paths = find_federated_core_paths(repo_root, algorithm)

    if not core_paths:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="federated_core_exists",
            status="failed",
            severity="failed",
            message="No federated core module found for algorithm.",
            path=(repo_root / "exaflow" / "algorithms" / "federated"),
            repo_root=repo_root,
            next_action=(
                "Create exaflow/algorithms/federated/<family>/"
                f"{algorithm}.py (or scaffold with --family and --with-federated-core)."
            ),
        )
        return

    if len(core_paths) > 1:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="federated_core_exists",
            status="failed",
            severity="failed",
            message="Multiple federated core modules found for same algorithm.",
            path=core_paths[0],
            repo_root=repo_root,
            next_action="Keep a single canonical federated core module.",
        )
        return

    core_path = core_paths[0]
    family = core_path.parent.name
    symbol = federated_symbol_for_algorithm(algorithm)

    register(
        report,
        algorithm=algorithm,
        phase="static",
        check="federated_core_exists",
        status="pass",
        severity="pass",
        message="Found federated core module.",
        path=core_path,
        repo_root=repo_root,
    )

    family_init = core_path.parent / "__init__.py"
    _check_token_in_file(
        report,
        algorithm=algorithm,
        repo_root=repo_root,
        path=family_init,
        check="family_init_registration",
        token=symbol,
        next_action=(
            f"Expose {symbol} in exaflow/algorithms/federated/{family}/__init__.py"
        ),
    )

    root_init = repo_root / "exaflow" / "algorithms" / "federated" / "__init__.py"
    _check_token_in_file(
        report,
        algorithm=algorithm,
        repo_root=repo_root,
        path=root_init,
        check="federated_root_registration",
        token=symbol,
        next_action="Expose federated symbol in exaflow/algorithms/federated/__init__.py",
    )

    specs_path = repo_root / "exaflow" / "algorithms" / "specifications.py"
    _check_token_in_file(
        report,
        algorithm=algorithm,
        repo_root=repo_root,
        path=specs_path,
        check="algorithm_name_enum_registration",
        token=f'"{algorithm}"',
        next_action=(
            "Add algorithm to AlgorithmName enum in "
            "exaflow/algorithms/specifications.py"
        ),
    )

    readme_path = repo_root / "exaflow" / "algorithms" / "federated" / "README.md"
    _check_token_in_file(
        report,
        algorithm=algorithm,
        repo_root=repo_root,
        path=readme_path,
        check="federated_readme_index",
        token=f"(docs/{algorithm}.md)",
        next_action=(
            "Add algorithm bullet under the family section in "
            "exaflow/algorithms/federated/README.md"
        ),
    )

    docs_path = (
        repo_root / "exaflow" / "algorithms" / "federated" / "docs" / f"{algorithm}.md"
    )
    if docs_path.exists():
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="federated_docs_exists",
            status="pass",
            severity="pass",
            message="Found federated docs entry.",
            path=docs_path,
            repo_root=repo_root,
        )
    else:
        register(
            report,
            algorithm=algorithm,
            phase="static",
            check="federated_docs_exists",
            status="failed",
            severity="failed",
            message="Missing federated docs markdown for algorithm.",
            path=docs_path,
            repo_root=repo_root,
            next_action=f"Create exaflow/algorithms/federated/docs/{algorithm}.md",
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

        prod_test = f"tests/prod_env_tests/test_{algorithm}.py"
        if (repo_root / prod_test).exists():
            generated.append(prod_test)

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
            ["uv", "run", "ruff", "check", "--select", "I", *lint_files],
            cwd=repo_root,
        )
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="ruff_check_select_I",
            status="pass" if rc == 0 else "failed",
            severity="pass" if rc == 0 else "failed",
            message="ruff check --select I passed."
            if rc == 0
            else (stderr or "ruff check failed."),
            path=None,
            repo_root=repo_root,
            next_action=(
                "Run: uv run ruff check --select I <files> and fix import-order issues."
            )
            if rc != 0
            else None,
        )

        rc, _, stderr = run_command(
            ["uv", "run", "ruff", "format", "--check", *lint_files],
            cwd=repo_root,
        )
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="ruff_format_check",
            status="pass" if rc == 0 else "failed",
            severity="pass" if rc == 0 else "failed",
            message="ruff format --check passed."
            if rc == 0
            else (stderr or "ruff format check failed."),
            path=None,
            repo_root=repo_root,
            next_action="Run: uv run ruff format <files>" if rc != 0 else None,
        )
    else:
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="ruff_checks",
            status="pass",
            severity="pass",
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
            ["uv", "run", "pytest", "--verbosity=2", *standalone_files],
            cwd=repo_root,
        )
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="standalone_tests",
            status="pass" if rc == 0 else "failed",
            severity="pass" if rc == 0 else "failed",
            message="Standalone tests passed."
            if rc == 0
            else (stderr or "Standalone tests failed."),
            path=None,
            repo_root=repo_root,
            next_action="Run standalone tests locally and fix failing assertions."
            if rc != 0
            else None,
        )
    else:
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="standalone_tests",
            status="failed",
            severity="failed",
            message="No standalone tests found for selected algorithms.",
            path=None,
            repo_root=repo_root,
            next_action=(
                "Create standalone tests under tests/standalone_tests/federated_algorithms/"
            ),
        )


def resolve_prod_test_file(repo_root: Path, algorithm: str) -> str | None:
    preferred = f"tests/prod_env_tests/test_{algorithm}.py"
    if (repo_root / preferred).exists():
        return preferred

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
            ["uv", "run", "pytest", "--verbosity=2", *prod_files],
            cwd=repo_root,
        )
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="prod_env_tests",
            status="pass" if rc == 0 else "failed",
            severity="pass" if rc == 0 else "failed",
            message="prod_env tests passed."
            if rc == 0
            else (stderr or "prod_env tests failed."),
            path=None,
            repo_root=repo_root,
            next_action="Run targeted prod_env tests and update expected fixtures/tests."
            if rc != 0
            else None,
        )
    else:
        register(
            report,
            algorithm="*",
            phase="runtime",
            check="prod_env_tests",
            status="failed",
            severity="failed",
            message="No prod_env tests found for selected algorithms.",
            path=None,
            repo_root=repo_root,
            next_action=(
                "Create canonical prod_env tests under tests/prod_env_tests/"
                "test_<algorithm>.py"
            ),
        )


def check_touched_registration_files(
    report: list[ReportEntry],
    *,
    algorithms: list[str],
    repo_root: Path,
    changed_files: list[str],
) -> None:
    touched = [
        rel
        for rel in changed_files
        if rel == "exaflow/algorithms/specifications.py"
        or rel == "exaflow/algorithms/federated/__init__.py"
        or re.fullmatch(r"exaflow/algorithms/federated/[a-z0-9_]+/__init__\.py", rel)
    ]

    if not touched:
        return

    expected_tokens: dict[str, set[str]] = {
        "exaflow/algorithms/specifications.py": set(),
        "exaflow/algorithms/federated/__init__.py": set(),
    }

    for algorithm in algorithms:
        expected_tokens["exaflow/algorithms/specifications.py"].add(f'"{algorithm}"')
        core_paths = find_federated_core_paths(repo_root, algorithm)
        if len(core_paths) == 1:
            family = core_paths[0].parent.name
            symbol = federated_symbol_for_algorithm(algorithm)
            family_init = f"exaflow/algorithms/federated/{family}/__init__.py"
            expected_tokens.setdefault(family_init, set()).add(symbol)
            expected_tokens["exaflow/algorithms/federated/__init__.py"].add(symbol)

    for rel in sorted(set(touched)):
        path = repo_root / rel
        if not path.exists():
            register(
                report,
                algorithm="*",
                phase="static",
                check="touched_registration_file_exists",
                status="failed",
                severity="failed",
                message="Touched registration file is missing.",
                path=path,
                repo_root=repo_root,
                next_action="Restore or recreate the touched registration file.",
            )
            continue

        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                register(
                    report,
                    algorithm="*",
                    phase="static",
                    check="touched_registration_file_syntax",
                    status="pass",
                    severity="pass",
                    message="Touched registration file parses successfully.",
                    path=path,
                    repo_root=repo_root,
                )
            except SyntaxError as exc:
                register(
                    report,
                    algorithm="*",
                    phase="static",
                    check="touched_registration_file_syntax",
                    status="failed",
                    severity="failed",
                    message=f"Syntax error: {exc}",
                    path=path,
                    repo_root=repo_root,
                    next_action="Fix Python syntax errors in touched registration file.",
                )

        tokens = expected_tokens.get(rel, set())
        if not tokens:
            continue

        content = path.read_text(encoding="utf-8")
        for token in sorted(tokens):
            if token in content:
                register(
                    report,
                    algorithm="*",
                    phase="static",
                    check="touched_registration_symbol_presence",
                    status="pass",
                    severity="pass",
                    message=f"Found expected token in touched file: {token}",
                    path=path,
                    repo_root=repo_root,
                )
            else:
                register(
                    report,
                    algorithm="*",
                    phase="static",
                    check="touched_registration_symbol_presence",
                    status="failed",
                    severity="failed",
                    message=f"Missing expected token in touched file: {token}",
                    path=path,
                    repo_root=repo_root,
                    next_action=(
                        "Re-run scaffold with registration patching or add missing "
                        "symbol manually."
                    ),
                )


def summarize(report: list[ReportEntry]) -> dict:
    rows = [entry.to_dict() for entry in report]
    failed = [row for row in rows if row["severity"] == "failed"]
    warnings = [row for row in rows if row["severity"] == "warn"]
    passed = [row for row in rows if row["severity"] == "pass"]
    return {
        "passed": passed,
        "warnings": warnings,
        "failed": failed,
        "report": rows,
    }


def _print_load_runtime_catalog_error(message: str) -> int:
    print(
        json.dumps(
            {
                "passed": [],
                "warnings": [],
                "failed": [
                    {
                        "algorithm": "*",
                        "phase": "static",
                        "check": "load_runtime_catalog",
                        "status": "failed",
                        "severity": "failed",
                        "message": message,
                        "next_action": (
                            "Run from repository root and ensure uv dependencies are "
                            "installed."
                        ),
                        "path": None,
                    }
                ],
                "report": [],
            },
            indent=2,
        )
    )
    return 1


def _print_arg_error(message: str) -> int:
    print(
        json.dumps(
            {
                "passed": [],
                "warnings": [],
                "failed": [
                    {
                        "algorithm": "*",
                        "phase": "static",
                        "check": "argument_validation",
                        "status": "failed",
                        "severity": "failed",
                        "message": message,
                        "next_action": (
                            "Adjust command arguments and re-run validator."
                        ),
                        "path": None,
                    }
                ],
                "report": [],
            },
            indent=2,
        )
    )
    return 1


def main() -> int:
    args = parse_args()
    repo_root = ensure_repo_root(args.repo_root)

    try:
        runtime_catalog = load_runtime_catalog(repo_root)
    except Exception as exc:  # pylint: disable=broad-except
        return _print_load_runtime_catalog_error(str(exc))

    changed_files = get_changed_files(repo_root)

    try:
        target_algorithms, new_algorithms, selection_mode = select_target_algorithms(
            args=args,
            runtime_catalog=runtime_catalog,
            changed_files=changed_files,
        )
    except ValueError as exc:
        return _print_arg_error(str(exc))

    report: list[ReportEntry] = []

    if not target_algorithms:
        register(
            report,
            algorithm="*",
            phase="static",
            check="target_selection",
            status="pass",
            severity="pass",
            message="No target algorithms detected. Nothing to validate.",
            path=None,
            repo_root=repo_root,
        )
        summary = summarize(report)
        summary["targets"] = target_algorithms
        summary["tier"] = "strict" if args.strict else "fast"
        summary["selection_mode"] = selection_mode
        summary["new_algorithm_targets"] = sorted(new_algorithms)
        print(json.dumps(summary, indent=2))
        return 0

    runtime_catalog_set = set(runtime_catalog)

    for algorithm in target_algorithms:
        is_new_mode = algorithm in new_algorithms

        check_import_and_spec(
            algorithm,
            report,
            repo_root=repo_root,
            runtime_catalog_set=runtime_catalog_set,
        )

        resolved_paths = check_required_paths(
            algorithm,
            report,
            repo_root=repo_root,
            enforce_canonical=is_new_mode,
        )

        module_path = (
            repo_root / "exaflow" / "algorithms" / "exareme3" / f"{algorithm}.py"
        )
        check_placeholder_file(
            algorithm,
            report,
            repo_root=repo_root,
            label="algorithm_module",
            path=module_path,
        )
        check_placeholder_file(
            algorithm,
            report,
            repo_root=repo_root,
            label="standalone_test",
            path=resolved_paths.standalone,
        )
        check_placeholder_file(
            algorithm,
            report,
            repo_root=repo_root,
            label="prod_test",
            path=resolved_paths.prod_test,
        )

        if is_new_mode:
            check_placeholder_file(
                algorithm,
                report,
                repo_root=repo_root,
                label="documentation",
                path=resolved_paths.documentation,
            )

        core_paths = find_federated_core_paths(repo_root, algorithm)
        if len(core_paths) == 1:
            check_placeholder_file(
                algorithm,
                report,
                repo_root=repo_root,
                label="federated_core",
                path=core_paths[0],
            )

        check_fixture_content(
            algorithm,
            report,
            repo_root=repo_root,
            fixture_path=resolved_paths.prod_expected,
            require_non_empty=is_new_mode,
        )

        if is_new_mode:
            check_new_algorithm_integration(
                algorithm,
                report,
                repo_root=repo_root,
            )
            check_placeholder_file(
                algorithm,
                report,
                repo_root=repo_root,
                label="federated_docs",
                path=(
                    repo_root
                    / "exaflow"
                    / "algorithms"
                    / "federated"
                    / "docs"
                    / f"{algorithm}.md"
                ),
            )

    check_touched_registration_files(
        report,
        algorithms=target_algorithms,
        repo_root=repo_root,
        changed_files=changed_files,
    )

    check_algorithm_profile_boundary(
        report,
        repo_root=repo_root,
        changed_files=changed_files,
    )

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
    summary["selection_mode"] = selection_mode
    summary["new_algorithm_targets"] = sorted(new_algorithms)
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
