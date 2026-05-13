#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ALGORITHM_ID_RE = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class GatePlan:
    name: str
    command: list[str] | None
    next_action: str | None = None


@dataclass
class GateResult:
    name: str
    status: str
    severity: str
    command: list[str] | None
    elapsed_seconds: float
    returncode: int | None
    message: str
    next_action: str | None
    stdout_tail: str | None = None
    stderr_tail: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "command": self.command,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "returncode": self.returncode,
            "message": self.message,
            "next_action": self.next_action,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the mechanical gates for a new Exaflow algorithm integration."
    )
    parser.add_argument("--algorithm", required=True, help="New algorithm identifier.")
    parser.add_argument(
        "--family",
        required=True,
        help="Federated family, for example statistics or linear_model.",
    )
    parser.add_argument(
        "--subfolder",
        help="Optional standalone-test subfolder. Defaults to --family.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the exaflow repository root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Include strict prod_env validator checks.",
    )
    parser.add_argument(
        "--skip-scaffold",
        action="store_true",
        help="Do not run scaffold; useful after implementation edits are already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned gate commands without executing them.",
    )
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each command gate.",
    )
    return parser.parse_args()


def ensure_repo_root(path: str) -> Path:
    repo_root = Path(path).resolve()
    if not (repo_root / "pyproject.toml").exists():
        raise ValueError(f"Invalid repo root: {repo_root}")
    if not (repo_root / "exaflow" / "algorithms" / "exareme3").exists():
        raise ValueError(f"Not an exaflow repository root: {repo_root}")
    return repo_root


def validate_identifier(value: str, *, label: str) -> str:
    if not ALGORITHM_ID_RE.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value}")
    return value


def expected_paths(
    repo_root: Path, algorithm: str, family: str, subfolder: str
) -> dict:
    return {
        "algorithm_module": repo_root
        / "exaflow"
        / "algorithms"
        / "exareme3"
        / f"{algorithm}.py",
        "federated_core": repo_root
        / "exaflow"
        / "algorithms"
        / "federated"
        / family
        / f"{algorithm}.py",
        "standalone_test": repo_root
        / "tests"
        / "standalone_tests"
        / "federated_algorithms"
        / subfolder
        / f"test_{algorithm}.py",
        "prod_test": repo_root / "tests" / "prod_env_tests" / f"test_{algorithm}.py",
        "prod_expected": repo_root
        / "tests"
        / "prod_env_tests"
        / "expected"
        / f"{algorithm}_expected.json",
    }


def rel_existing_python_files(paths: dict, repo_root: Path) -> list[str]:
    candidates = [
        paths["algorithm_module"],
        paths["federated_core"],
        paths["standalone_test"],
        paths["prod_test"],
    ]
    return sorted(
        str(path.relative_to(repo_root))
        for path in candidates
        if path.exists() and path.suffix == ".py"
    )


def build_gate_plan(
    *,
    algorithm: str,
    family: str,
    subfolder: str,
    repo_root: Path,
    strict: bool,
    skip_scaffold: bool,
) -> list[GatePlan]:
    paths = expected_paths(repo_root, algorithm, family, subfolder)
    lint_files = rel_existing_python_files(paths, repo_root)
    standalone_rel = str(paths["standalone_test"].relative_to(repo_root))

    gates: list[GatePlan] = []
    if not skip_scaffold:
        gates.append(
            GatePlan(
                name="scaffold",
                command=[
                    "poetry",
                    "run",
                    "python",
                    ".agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py",
                    "--repo-root",
                    ".",
                    "--algorithms",
                    algorithm,
                    "--family",
                    family,
                    "--subfolder",
                    subfolder,
                ],
                next_action="Fix scaffold errors before editing generated files.",
            )
        )

    validate_command = [
        "poetry",
        "run",
        "python",
        ".agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py",
        "--repo-root",
        ".",
        "--new-algorithm",
        algorithm,
    ]
    if strict:
        validate_command.append("--strict")
    gates.append(
        GatePlan(
            name="validate_new_algorithm",
            command=validate_command,
            next_action="Resolve every failed or warning row in validator JSON.",
        )
    )

    if lint_files:
        gates.append(
            GatePlan(
                name="ruff_import_order",
                command=[
                    "poetry",
                    "run",
                    "ruff",
                    "check",
                    "--select",
                    "I",
                    *lint_files,
                ],
                next_action="Run ruff with --fix or sort imports manually.",
            )
        )
        gates.append(
            GatePlan(
                name="ruff_format",
                command=["poetry", "run", "ruff", "format", "--check", *lint_files],
                next_action="Run poetry run ruff format on the listed files.",
            )
        )
    else:
        gates.append(
            GatePlan(
                name="ruff_checks",
                command=None,
                next_action="Run scaffold or create implementation/test files first.",
            )
        )

    gates.append(
        GatePlan(
            name="standalone_pytest",
            command=["poetry", "run", "pytest", "-q", standalone_rel]
            if paths["standalone_test"].exists()
            else None,
            next_action=f"Create and implement {standalone_rel}.",
        )
    )
    return gates


def _tail(text: str | None, *, limit: int = 2000) -> str | None:
    if not text:
        return None
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def run_gate(plan: GatePlan, *, cwd: Path, timeout: int) -> GateResult:
    if plan.command is None:
        return GateResult(
            name=plan.name,
            status="skipped",
            severity="failed",
            command=None,
            elapsed_seconds=0.0,
            returncode=None,
            message="Required files are missing, so this gate cannot run.",
            next_action=plan.next_action,
        )

    started = time.monotonic()
    try:
        proc = subprocess.run(
            plan.command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return GateResult(
            name=plan.name,
            status="timeout",
            severity="failed",
            command=plan.command,
            elapsed_seconds=time.monotonic() - started,
            returncode=None,
            message=f"Command exceeded {timeout}s timeout.",
            next_action="Interrupt the integration and report the timed-out gate.",
            stdout_tail=_tail(exc.stdout),
            stderr_tail=_tail(exc.stderr),
        )

    elapsed = time.monotonic() - started
    severity = "pass" if proc.returncode == 0 else "failed"
    message = "Gate passed." if proc.returncode == 0 else "Gate failed."

    payload = extract_json_object(proc.stdout)
    if plan.name == "validate_new_algorithm" and payload is not None:
        warnings = payload.get("warnings") or []
        failed = payload.get("failed") or []
        if failed:
            severity = "failed"
            message = f"Validator reported {len(failed)} failed rows."
        elif warnings:
            severity = "failed"
            message = f"Validator reported {len(warnings)} warning rows."

    return GateResult(
        name=plan.name,
        status="pass" if severity == "pass" else "failed",
        severity=severity,
        command=plan.command,
        elapsed_seconds=elapsed,
        returncode=proc.returncode,
        message=message,
        next_action=None if severity == "pass" else plan.next_action,
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
    )


def planned_report(
    *,
    algorithm: str,
    family: str,
    strict: bool,
    gates: list[GatePlan],
) -> dict:
    return {
        "algorithm": algorithm,
        "family": family,
        "strict": strict,
        "done": False,
        "mode": "dry-run",
        "gates": [
            {
                "name": gate.name,
                "status": "planned",
                "severity": "pass",
                "command": gate.command,
                "next_action": gate.next_action,
            }
            for gate in gates
        ],
    }


def summarize(
    *,
    algorithm: str,
    family: str,
    strict: bool,
    gates: list[GateResult],
    elapsed_seconds: float,
) -> dict:
    failed = [gate.to_dict() for gate in gates if gate.severity == "failed"]
    passed = [gate.to_dict() for gate in gates if gate.severity == "pass"]
    return {
        "algorithm": algorithm,
        "family": family,
        "strict": strict,
        "done": not failed,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "passed": passed,
        "failed": failed,
        "gates": [gate.to_dict() for gate in gates],
    }


def main() -> int:
    args = parse_args()
    repo_root = ensure_repo_root(args.repo_root)
    algorithm = validate_identifier(args.algorithm, label="algorithm")
    family = validate_identifier(args.family, label="family")
    subfolder = validate_identifier(args.subfolder or family, label="subfolder")

    gates = build_gate_plan(
        algorithm=algorithm,
        family=family,
        subfolder=subfolder,
        repo_root=repo_root,
        strict=args.strict,
        skip_scaffold=args.skip_scaffold,
    )

    if args.dry_run:
        print(
            json.dumps(
                planned_report(
                    algorithm=algorithm,
                    family=family,
                    strict=args.strict,
                    gates=gates,
                ),
                indent=2,
            )
        )
        return 0

    started = time.monotonic()
    results = [
        run_gate(gate, cwd=repo_root, timeout=args.command_timeout) for gate in gates
    ]
    summary = summarize(
        algorithm=algorithm,
        family=family,
        strict=args.strict,
        gates=results,
        elapsed_seconds=time.monotonic() - started,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
