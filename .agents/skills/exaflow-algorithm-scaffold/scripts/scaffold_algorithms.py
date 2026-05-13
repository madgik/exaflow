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
        "levene_test",
        "pearson_correlation",
        "ttest_independent",
        "ttest_onesample",
        "ttest_paired",
    },
}

FAMILY_PREFIX_RULES = {
    "naive_bayes": "naive_bayes",
    "ttest": "statistics",
    "anova": "statistics",
    "levene": "statistics",
    "pearson": "statistics",
    "linear_regression": "linear_model",
    "logistic_regression": "linear_model",
    "linear_svm": "linear_model",
    "pca": "decomposition",
    "kmeans": "cluster",
}


@dataclass(frozen=True)
class InputSpec:
    label: str
    desc: str
    types: tuple[str, ...]
    stattypes: tuple[str, ...]
    required: bool
    multiple: bool


@dataclass(frozen=True)
class FamilyProfile:
    desc: str
    label: str
    y: InputSpec
    x: InputSpec | None
    parameters_literal: str
    components: tuple[str, ...]
    with_aggregation_server: bool


FAMILY_PROFILES = {
    "statistics": FamilyProfile(
        desc=(
            "Federated statistical workflow placeholder. Replace the placeholder "
            "logic with concrete federated computations."
        ),
        label="Statistics Placeholder",
        y=InputSpec(
            label="Value variable",
            desc="Single numerical variable used by the statistical procedure.",
            types=("REAL", "INT"),
            stattypes=("NUMERICAL",),
            required=True,
            multiple=False,
        ),
        x=InputSpec(
            label="Grouping variable",
            desc="Single nominal variable defining groups or strata.",
            types=("TEXT",),
            stattypes=("NOMINAL",),
            required=True,
            multiple=False,
        ),
        parameters_literal="{}",
        components=("AGGREGATION_SERVER",),
        with_aggregation_server=True,
    ),
    "linear_model": FamilyProfile(
        desc=(
            "Federated linear-model placeholder. Replace with model-specific "
            "estimation logic and validation."
        ),
        label="Linear Model Placeholder",
        y=InputSpec(
            label="Outcome variable",
            desc="Single dependent variable.",
            types=("REAL", "INT"),
            stattypes=("NUMERICAL",),
            required=True,
            multiple=False,
        ),
        x=InputSpec(
            label="Covariates",
            desc="One or more covariates (numerical and/or categorical).",
            types=("REAL", "INT", "TEXT"),
            stattypes=("NUMERICAL", "NOMINAL"),
            required=True,
            multiple=True,
        ),
        parameters_literal="{}",
        components=("AGGREGATION_SERVER",),
        with_aggregation_server=True,
    ),
    "decomposition": FamilyProfile(
        desc=(
            "Federated decomposition placeholder. Replace with the target "
            "decomposition algorithm implementation."
        ),
        label="Decomposition Placeholder",
        y=InputSpec(
            label="Feature variables",
            desc="One or more numerical features for decomposition.",
            types=("REAL", "INT"),
            stattypes=("NUMERICAL",),
            required=True,
            multiple=True,
        ),
        x=None,
        parameters_literal="{}",
        components=("AGGREGATION_SERVER",),
        with_aggregation_server=True,
    ),
    "naive_bayes": FamilyProfile(
        desc=(
            "Federated Naive Bayes placeholder. Replace with concrete probabilistic "
            "model fitting and inference."
        ),
        label="Naive Bayes Placeholder",
        y=InputSpec(
            label="Target variable",
            desc="Single nominal target label.",
            types=("TEXT",),
            stattypes=("NOMINAL",),
            required=True,
            multiple=False,
        ),
        x=InputSpec(
            label="Feature variables",
            desc="One or more feature variables.",
            types=("REAL", "INT", "TEXT"),
            stattypes=("NUMERICAL", "NOMINAL"),
            required=True,
            multiple=True,
        ),
        parameters_literal="{}",
        components=("AGGREGATION_SERVER",),
        with_aggregation_server=True,
    ),
    "cluster": FamilyProfile(
        desc=(
            "Federated clustering placeholder. Replace with the target clustering "
            "workflow and convergence checks."
        ),
        label="Cluster Placeholder",
        y=InputSpec(
            label="Feature variables",
            desc="One or more numerical features to cluster.",
            types=("REAL", "INT"),
            stattypes=("NUMERICAL",),
            required=True,
            multiple=True,
        ),
        x=None,
        parameters_literal="{}",
        components=("AGGREGATION_SERVER",),
        with_aggregation_server=True,
    ),
}

DEFAULT_PROFILE = FamilyProfile(
    desc="TODO: describe the algorithm behavior and assumptions.",
    label="Generated Algorithm",
    y=InputSpec(
        label="Dependent variable",
        desc="TODO: define dependent variable constraints.",
        types=("REAL", "INT"),
        stattypes=("NUMERICAL",),
        required=True,
        multiple=False,
    ),
    x=InputSpec(
        label="Independent variables",
        desc="TODO: define independent variable constraints.",
        types=("REAL", "INT"),
        stattypes=("NUMERICAL",),
        required=True,
        multiple=True,
    ),
    parameters_literal="{}",
    components=(),
    with_aggregation_server=False,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold Exaflow algorithm artifacts with optional integration patching."
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
        "--family",
        help="Optional federated family (for example: statistics, linear_model).",
    )
    parser.add_argument(
        "--subfolder",
        help=(
            "Explicit tests/standalone_tests/federated_algorithms subfolder. "
            "Takes precedence over --family inference."
        ),
    )
    parser.add_argument(
        "--with-federated-core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create federated core placeholder module under exaflow/algorithms/federated.",
    )
    parser.add_argument(
        "--with-registration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Patch common integration touchpoints (family/root __init__, "
            "AlgorithmName enum)."
        ),
    )
    parser.add_argument(
        "--with-doc-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch exaflow/algorithms/federated/README.md index and docs stub.",
    )
    parser.add_argument(
        "--with-sample-fixture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a non-empty sample prod fixture instead of an empty test_cases array.",
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
    invalid = [name for name in parts if not ALGORITHM_ID_RE.fullmatch(name)]
    if invalid:
        raise ValueError(
            f"Invalid algorithm identifiers: {', '.join(sorted(set(invalid)))}"
        )
    return sorted(set(parts))


def parse_family(raw: str | None) -> str | None:
    if not raw:
        return None
    family = raw.strip()
    if not ALGORITHM_ID_RE.fullmatch(family):
        raise ValueError(f"Invalid family identifier: '{raw}'")
    return family


def parse_subfolder(raw: str | None) -> str | None:
    if not raw:
        return None
    subfolder = raw.strip().strip("/")
    if not ALGORITHM_ID_RE.fullmatch(subfolder):
        raise ValueError(f"Invalid subfolder identifier: '{raw}'")
    return subfolder


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


def snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def federated_symbol_for_algorithm(algorithm: str) -> str:
    return f"Federated{snake_to_pascal(algorithm)}"


def infer_family_from_algorithm(algorithm: str) -> str | None:
    for family, algorithms in STANDALONE_FOLDER_RULES.items():
        if algorithm in algorithms:
            return family

    for prefix, family in FAMILY_PREFIX_RULES.items():
        if algorithm.startswith(prefix):
            return family

    return None


def resolve_family_and_subfolder(
    algorithm: str,
    *,
    family_override: str | None,
    subfolder_override: str | None,
) -> tuple[str | None, str, str]:
    family = family_override or infer_family_from_algorithm(algorithm)

    if subfolder_override:
        return family, subfolder_override, "explicit"
    if family:
        return family, family, "family"
    return None, "_generated", "fallback"


def profile_for_family(family: str | None) -> FamilyProfile:
    if family is None:
        return DEFAULT_PROFILE
    return FAMILY_PROFILES.get(family, DEFAULT_PROFILE)


def _input_spec_literal(spec: InputSpec, *, key: str) -> str:
    types_literal = ", ".join(f"specs.InputDataType.{value}" for value in spec.types)
    stattypes_literal = ", ".join(
        f"specs.InputDataStatType.{value}" for value in spec.stattypes
    )
    return (
        f"                {key}=specs.InputDataSpecification(\n"
        f"                    label={json.dumps(spec.label)},\n"
        f"                    desc={json.dumps(spec.desc)},\n"
        f"                    types=[{types_literal}],\n"
        f"                    stattypes=[{stattypes_literal}],\n"
        f"                    required={spec.required},\n"
        f"                    multiple={spec.multiple},\n"
        "                    enumslen=None,\n"
        "                ),"
    )


def _components_literal(profile: FamilyProfile) -> str:
    if not profile.components:
        return "[]"
    parts = ", ".join(f"specs.ComponentType.{item}" for item in profile.components)
    return f"[{parts}]"


def algorithm_template(
    algorithm: str,
    *,
    profile: FamilyProfile,
    family: str | None,
    include_federated_core: bool,
) -> str:
    class_name = snake_to_pascal(algorithm)
    result_class = f"{class_name}Result"
    federated_symbol = federated_symbol_for_algorithm(algorithm)

    imports = [
        "from pydantic import BaseModel",
        "",
        "from exaflow.algorithms import specifications as specs",
        "from exaflow.algorithms.exareme3.utils.algorithm import Algorithm",
        "from exaflow.algorithms.exareme3.utils.registry import exareme3_udf",
    ]

    if include_federated_core and family:
        imports.append(
            f"from exaflow.algorithms.federated.{family}.{algorithm} import {federated_symbol}"
        )

    y_literal = _input_spec_literal(profile.y, key="y")
    x_literal = (
        _input_spec_literal(profile.x, key="x")
        if profile.x is not None
        else "                x=None,"
    )

    local_step_logic = (
        f"    estimator = {federated_symbol}(agg_client=agg_client)\n"
        "    return estimator.compute(\n"
        "        data=data,\n"
        "        target_var=target_var,\n"
        "        feature_vars=feature_vars,\n"
        "        metadata=metadata,\n"
        "    )"
        if include_federated_core and family
        else (
            "    raise NotImplementedError(\n"
            f'        "TODO: implement UDF logic for {algorithm}."\n'
            "    )"
        )
    )

    return (
        "\n".join(imports)
        + f'''


class {result_class}(BaseModel):
    result: dict


class {class_name}(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="{algorithm}",
            desc={json.dumps(profile.desc)},
            label={json.dumps(profile.label if family else snake_to_title(algorithm))},
            enabled=True,
            inputdata=specs.InputDataSpecifications(
{y_literal}
{x_literal}
                validation=None,
            ),
            parameters={profile.parameters_literal},
            type=specs.AlgorithmType.EXAREME3,
            components={_components_literal(profile)},
        )

    def run(self):
        result = self.run_local_udf(
            func=local_step,
            kw_args={{
                "target_var": self.inputdata.y[0],
                "feature_vars": list(self.inputdata.x) if self.inputdata.x else [],
            }},
            identical_results=True,
        )
        return {result_class}(result=result)


@exareme3_udf(with_aggregation_server={profile.with_aggregation_server})
def local_step(agg_client, data, target_var, feature_vars, metadata):
{local_step_logic}
'''
    )


def federated_core_template(algorithm: str) -> str:
    federated_symbol = federated_symbol_for_algorithm(algorithm)
    return f'''from __future__ import annotations


class {federated_symbol}:
    """Federated core placeholder for `{algorithm}`."""

    def __init__(self, agg_client=None):
        self.agg_client = agg_client

    def compute(self, *, data, target_var, feature_vars, metadata):
        raise NotImplementedError(
            "TODO: replace placeholder with federated runtime logic."
        )
'''


def standalone_test_template(algorithm: str) -> str:
    return f"""def test_{algorithm}_standalone_placeholder():
    raise NotImplementedError(
        "TODO: replace placeholder with federated standalone tests for {algorithm}."
    )
"""


def prod_validation_test_template(algorithm: str) -> str:
    return f'''from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

algorithm_name = "{algorithm}"
expected_file = Path(__file__).parent / "expected" / f"{{algorithm_name}}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_{algorithm}_validation(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)
    assert result
'''


def prod_expected_template(*, include_sample_fixture: bool) -> str:
    if not include_sample_fixture:
        payload = {"test_cases": []}
        return json.dumps(payload, indent=2) + "\n"

    payload = {
        "test_cases": [
            {
                "input": {
                    "inputdata": {
                        "y": ["__REPLACE_ME_Y__"],
                        "x": ["__REPLACE_ME_X__"],
                        "data_model": "__REPLACE_ME_DATA_MODEL__",
                        "datasets": ["__REPLACE_ME_DATASET__"],
                        "filters": None,
                    },
                    "parameters": {},
                },
                "output": {},
            }
        ]
    }
    return json.dumps(payload, indent=2) + "\n"


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


def federated_docs_template(algorithm: str, family: str) -> str:
    return f"""# {snake_to_title(algorithm)}

## Family

- `{family}`

## Purpose

TODO: describe federated core behavior for `{algorithm}`.

## Notes

- Generated scaffold placeholder.
- Replace with concrete algorithm-specific documentation.
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
    severity: str,
    message: str,
    path: Path | None,
    repo_root: Path,
    next_action: str | None = None,
) -> None:
    report.append(
        ReportEntry(
            algorithm=algorithm,
            phase="scaffold",
            check=check,
            status=status,
            severity=severity,
            message=message,
            next_action=next_action,
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
) -> bool:
    if path.exists():
        register(
            report,
            algorithm=algorithm,
            check=check,
            status="skipped_existing",
            severity="pass",
            message="File already exists; no overwrite performed.",
            path=path,
            repo_root=repo_root,
        )
        return False

    if dry_run:
        register(
            report,
            algorithm=algorithm,
            check=check,
            status="created",
            severity="pass",
            message="Dry-run: file would be created.",
            path=path,
            repo_root=repo_root,
        )
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    register(
        report,
        algorithm=algorithm,
        check=check,
        status="created",
        severity="pass",
        message="File created.",
        path=path,
        repo_root=repo_root,
    )
    return True


def _format_all_block(items: list[str]) -> str:
    lines = ["__all__ = ["]
    lines.extend(f'    "{item}",' for item in items)
    lines.append("]")
    return "\n".join(lines)


def _upsert_symbol_in_init(
    content: str,
    *,
    import_line: str,
    symbol: str,
) -> tuple[str, bool]:
    text = content
    changed = False

    if import_line not in text:
        all_match = re.search(r"^__all__\s*=", text, flags=re.MULTILINE)
        if all_match:
            text = (
                text[: all_match.start()]
                + import_line
                + "\n"
                + text[all_match.start() :]
            )
        else:
            text = text.rstrip() + "\n" + import_line + "\n"
        changed = True

    all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
    if all_match:
        items = re.findall(r"['\"]([^'\"]+)['\"]", all_match.group(1))
        if symbol not in items:
            items.append(symbol)
            new_block = _format_all_block(items)
            text = text[: all_match.start()] + new_block + text[all_match.end() :]
            changed = True
    else:
        block = _format_all_block([symbol])
        if text.strip():
            text = text.rstrip() + "\n\n" + block + "\n"
        else:
            text = block + "\n"
        changed = True

    if not text.endswith("\n"):
        text += "\n"
    return text, changed


def _patch_algorithm_enum(content: str, algorithm: str) -> tuple[str, bool, str]:
    if f'"{algorithm}"' in content:
        return content, False, "Algorithm already present in AlgorithmName enum."

    class_match = re.search(
        r"^class AlgorithmName\(str, Enum\):$", content, re.MULTILINE
    )
    if not class_match:
        raise ValueError("Could not find AlgorithmName enum in specifications.py")

    insert_anchor = content.find("\n    def __str__(self) -> str:", class_match.start())
    if insert_anchor == -1:
        raise ValueError("Could not find insertion point in AlgorithmName enum")

    enum_line = f'    {algorithm.upper()} = "{algorithm}"\n'
    updated = content[:insert_anchor] + enum_line + content[insert_anchor:]
    return updated, True, "Algorithm added to AlgorithmName enum."


def _patch_federated_readme(
    content: str, family: str, algorithm: str
) -> tuple[str, bool]:
    lines = content.splitlines()
    header = f"## {family}"
    bullet = f"- [{algorithm.upper()}](docs/{algorithm}.md)"

    try:
        section_start = lines.index(header)
    except ValueError:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([header, bullet])
        return "\n".join(lines) + "\n", True

    section_end = len(lines)
    for idx in range(section_start + 1, len(lines)):
        if lines[idx].startswith("## "):
            section_end = idx
            break

    if any(line.strip() == bullet for line in lines[section_start:section_end]):
        return content, False

    lines.insert(section_end, bullet)
    return "\n".join(lines) + "\n", True


def patch_text_file(
    report: list[ReportEntry],
    *,
    algorithm: str,
    check: str,
    path: Path,
    repo_root: Path,
    dry_run: bool,
    patcher,
    missing_content: str | None = None,
    next_action: str | None = None,
) -> None:
    if not path.exists() and missing_content is None:
        register(
            report,
            algorithm=algorithm,
            check=check,
            status="failed",
            severity="failed",
            message="Target file does not exist.",
            path=path,
            repo_root=repo_root,
            next_action=next_action,
        )
        return

    original = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else missing_content
        if missing_content is not None
        else ""
    )

    try:
        updated, changed, message = patcher(original)
    except Exception as exc:  # pylint: disable=broad-except
        register(
            report,
            algorithm=algorithm,
            check=check,
            status="failed",
            severity="failed",
            message=str(exc),
            path=path,
            repo_root=repo_root,
            next_action=next_action,
        )
        return

    if not changed:
        register(
            report,
            algorithm=algorithm,
            check=check,
            status="skipped_existing",
            severity="pass",
            message=message,
            path=path,
            repo_root=repo_root,
        )
        return

    if dry_run:
        register(
            report,
            algorithm=algorithm,
            check=check,
            status="patched",
            severity="pass",
            message=f"Dry-run: {message}",
            path=path,
            repo_root=repo_root,
        )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    register(
        report,
        algorithm=algorithm,
        check=check,
        status="patched",
        severity="pass",
        message=message,
        path=path,
        repo_root=repo_root,
    )


def run_for_algorithm(
    algorithm: str,
    report: list[ReportEntry],
    *,
    repo_root: Path,
    dry_run: bool,
    family_override: str | None,
    subfolder_override: str | None,
    with_federated_core: bool,
    with_registration: bool,
    with_doc_index: bool,
    with_sample_fixture: bool,
    runtime_catalog_set: set[str],
) -> None:
    family, standalone_subfolder, resolution_source = resolve_family_and_subfolder(
        algorithm,
        family_override=family_override,
        subfolder_override=subfolder_override,
    )

    if resolution_source == "fallback":
        register(
            report,
            algorithm=algorithm,
            check="standalone_subfolder_resolution",
            status="subfolder_fallback",
            severity="warn",
            message=(
                "Could not infer standalone subfolder from algorithm/family; "
                "falling back to _generated."
            ),
            path=(
                repo_root
                / "tests"
                / "standalone_tests"
                / "federated_algorithms"
                / "_generated"
            ),
            repo_root=repo_root,
            next_action="Re-run with --subfolder <family> or --family <family>.",
        )

    profile = profile_for_family(family)

    family_path = (
        repo_root / "exaflow" / "algorithms" / "federated" / family
        if family is not None
        else None
    )
    can_create_federated_core = (
        with_federated_core and family is not None and family_path is not None
    )

    if with_federated_core and family is None:
        register(
            report,
            algorithm=algorithm,
            check="federated_family_resolution",
            status="skipped_unknown_family",
            severity="warn",
            message="Federated core creation skipped because no family could be inferred.",
            path=None,
            repo_root=repo_root,
            next_action="Provide --family to generate federated core artifacts.",
        )

    module_path = repo_root / "exaflow" / "algorithms" / "exareme3" / f"{algorithm}.py"
    standalone_test = (
        repo_root
        / "tests"
        / "standalone_tests"
        / "federated_algorithms"
        / standalone_subfolder
        / f"test_{algorithm}.py"
    )
    prod_test = repo_root / "tests" / "prod_env_tests" / f"test_{algorithm}.py"
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
        content=algorithm_template(
            algorithm,
            profile=profile,
            family=family,
            include_federated_core=can_create_federated_core,
        ),
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
        content=prod_expected_template(include_sample_fixture=with_sample_fixture),
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

    federated_core_path: Path | None = None
    if can_create_federated_core and family_path is not None:
        federated_core_path = family_path / f"{algorithm}.py"
        write_placeholder(
            report,
            algorithm=algorithm,
            check="federated_core_module",
            path=federated_core_path,
            content=federated_core_template(algorithm),
            dry_run=dry_run,
            repo_root=repo_root,
        )

    if can_create_federated_core and with_doc_index and family is not None:
        federated_doc_path = (
            repo_root
            / "exaflow"
            / "algorithms"
            / "federated"
            / "docs"
            / f"{algorithm}.md"
        )
        write_placeholder(
            report,
            algorithm=algorithm,
            check="federated_doc_stub",
            path=federated_doc_path,
            content=federated_docs_template(algorithm, family),
            dry_run=dry_run,
            repo_root=repo_root,
        )

    is_new_algorithm = algorithm not in runtime_catalog_set
    if not is_new_algorithm:
        register(
            report,
            algorithm=algorithm,
            check="new_algorithm_detection",
            status="existing_algorithm",
            severity="pass",
            message="Algorithm already present in runtime catalog; registration patching skipped.",
            path=None,
            repo_root=repo_root,
        )
        return

    if not can_create_federated_core or family is None:
        return

    if with_registration:
        family_init_path = family_path / "__init__.py"
        federated_symbol = federated_symbol_for_algorithm(algorithm)

        patch_text_file(
            report,
            algorithm=algorithm,
            check="patch_family_init",
            path=family_init_path,
            repo_root=repo_root,
            dry_run=dry_run,
            patcher=lambda content: (
                *_upsert_symbol_in_init(
                    content,
                    import_line=f"from .{algorithm} import {federated_symbol}",
                    symbol=federated_symbol,
                ),
                "Family __init__.py updated with import and __all__ entry.",
            ),
            missing_content="",
            next_action=f"Add import and __all__ entry for {federated_symbol}.",
        )

        root_federated_init = (
            repo_root / "exaflow" / "algorithms" / "federated" / "__init__.py"
        )
        patch_text_file(
            report,
            algorithm=algorithm,
            check="patch_federated_root_init",
            path=root_federated_init,
            repo_root=repo_root,
            dry_run=dry_run,
            patcher=lambda content: (
                *_upsert_symbol_in_init(
                    content,
                    import_line=(
                        f"from exaflow.algorithms.federated.{family} "
                        f"import {federated_symbol}"
                    ),
                    symbol=federated_symbol,
                ),
                "Federated root __init__.py updated with import and __all__ entry.",
            ),
            next_action=(
                "Expose the federated symbol from "
                "exaflow/algorithms/federated/__init__.py."
            ),
        )

        specs_path = repo_root / "exaflow" / "algorithms" / "specifications.py"
        patch_text_file(
            report,
            algorithm=algorithm,
            check="patch_algorithm_name_enum",
            path=specs_path,
            repo_root=repo_root,
            dry_run=dry_run,
            patcher=lambda content: _patch_algorithm_enum(content, algorithm),
            next_action=(
                "Add the algorithm to AlgorithmName enum in "
                "exaflow/algorithms/specifications.py."
            ),
        )

    if with_doc_index:
        federated_readme = (
            repo_root / "exaflow" / "algorithms" / "federated" / "README.md"
        )
        patch_text_file(
            report,
            algorithm=algorithm,
            check="patch_federated_doc_index",
            path=federated_readme,
            repo_root=repo_root,
            dry_run=dry_run,
            patcher=lambda content: (
                *_patch_federated_readme(content, family, algorithm),
                "Federated README index updated.",
            ),
            next_action=(
                "Add algorithm bullet under the family section in "
                "exaflow/algorithms/federated/README.md."
            ),
        )


def summarize(report: Iterable[ReportEntry]) -> dict:
    rows = [entry.to_dict() for entry in report]
    created = [row for row in rows if row["status"] == "created"]
    patched = [row for row in rows if row["status"] == "patched"]
    skipped = [row for row in rows if row["status"].startswith("skipped")]
    warnings = [row for row in rows if row["severity"] == "warn"]
    failed = [row for row in rows if row["severity"] == "failed"]
    return {
        "created": created,
        "patched": patched,
        "skipped": skipped,
        "warnings": warnings,
        "failed": failed,
        "report": rows,
    }


def _print_target_selection_error(message: str) -> int:
    print(
        json.dumps(
            {
                "created": [],
                "patched": [],
                "skipped": [],
                "warnings": [],
                "failed": [
                    {
                        "algorithm": "*",
                        "phase": "scaffold",
                        "check": "target_selection",
                        "status": "failed",
                        "severity": "failed",
                        "message": message,
                        "next_action": "Provide --algorithms <name> or --all.",
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

    family_override = parse_family(args.family)
    subfolder_override = parse_subfolder(args.subfolder)

    if not args.algorithms and not args.all:
        return _print_target_selection_error(
            "No target algorithms provided. Use --algorithms or --all."
        )

    runtime_catalog: list[str] = []
    runtime_catalog_error: str | None = None

    if args.all or args.with_registration or args.with_federated_core:
        try:
            runtime_catalog = load_runtime_catalog(repo_root)
        except Exception as exc:  # pylint: disable=broad-except
            runtime_catalog_error = str(exc)
            if args.all:
                print(
                    json.dumps(
                        {
                            "created": [],
                            "patched": [],
                            "skipped": [],
                            "warnings": [],
                            "failed": [
                                {
                                    "algorithm": "*",
                                    "phase": "scaffold",
                                    "check": "load_runtime_catalog",
                                    "status": "failed",
                                    "severity": "failed",
                                    "message": runtime_catalog_error,
                                    "next_action": (
                                        "Run from the repository root and ensure "
                                        "poetry dependencies are available."
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

    if args.algorithms:
        target_algorithms = parse_algorithm_list(args.algorithms)
    else:
        target_algorithms = runtime_catalog

    if not target_algorithms:
        return _print_target_selection_error("No target algorithms resolved.")

    report: list[ReportEntry] = []
    if runtime_catalog_error and args.algorithms:
        register(
            report,
            algorithm="*",
            check="load_runtime_catalog",
            status="runtime_catalog_unavailable",
            severity="warn",
            message=runtime_catalog_error,
            path=None,
            repo_root=repo_root,
            next_action=(
                "Registration/new-algorithm detection is best-effort without runtime "
                "catalog; install dependencies for full behavior."
            ),
        )

    runtime_catalog_set = set(runtime_catalog)

    for algorithm in target_algorithms:
        try:
            run_for_algorithm(
                algorithm,
                report,
                repo_root=repo_root,
                dry_run=args.dry_run,
                family_override=family_override,
                subfolder_override=subfolder_override,
                with_federated_core=args.with_federated_core,
                with_registration=args.with_registration,
                with_doc_index=args.with_doc_index,
                with_sample_fixture=args.with_sample_fixture,
                runtime_catalog_set=runtime_catalog_set,
            )
        except Exception as exc:  # pylint: disable=broad-except
            register(
                report,
                algorithm=algorithm,
                check="scaffold_algorithm",
                status="failed",
                severity="failed",
                message=str(exc),
                path=None,
                repo_root=repo_root,
                next_action="Fix the reported error and re-run the scaffold command.",
            )

    summary = summarize(report)
    summary["mode"] = "dry-run" if args.dry_run else "apply"
    summary["targets"] = target_algorithms
    summary["options"] = {
        "family": family_override,
        "subfolder": subfolder_override,
        "with_federated_core": args.with_federated_core,
        "with_registration": args.with_registration,
        "with_doc_index": args.with_doc_index,
        "with_sample_fixture": args.with_sample_fixture,
    }
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
