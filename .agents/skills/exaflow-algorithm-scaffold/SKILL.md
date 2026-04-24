---
name: exaflow-algorithm-scaffold
description: "Scaffold and fully integrate new Exaflow Exareme3 algorithms end-to-end. Use when asked to add, create, implement, or integrate a new algorithm (for example Levene's test), including code, tests, docs, and validation."
---

# Exaflow Algorithm Scaffold

Use this skill for end-to-end algorithm integration. Do not stop at placeholders unless the user explicitly asks for placeholder-only output.

## Default Execution Policy

1. Scaffold missing files for the target algorithm.
2. Implement algorithm logic in `exaflow/algorithms/exareme3/<algorithm>.py`.
3. Implement standalone test coverage in `tests/standalone_tests/federated_algorithms/...`.
4. Implement prod validation test and expected fixture in `tests/prod_env_tests`.
5. Implement or update algorithm docs under `documentation/algorithms/`.
6. Run validation with `$exaflow-algorithm-validate` (fast, then strict when requested).

## Commands

Scaffold all missing artifacts for one algorithm:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_new_algorithm
```

Plan-only / impact preview:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_new_algorithm --dry-run
```

## Behavior Contract

- Source of truth for `--all`: runtime Exareme3 algorithm catalog from `exaflow.exareme3_algorithm_classes`.
- `--algorithms` accepts new algorithm IDs not yet present in runtime catalog.
- Overwrite policy: never overwrite existing files.
- Standalone test placement: infer `tests/standalone_tests/federated_algorithms/<subfolder>/test_<algorithm>.py`; fallback to `_generated`.
- Created placeholders:
  - `exaflow/algorithms/exareme3/<algorithm>.py`
  - `tests/standalone_tests/federated_algorithms/<subfolder>/test_<algorithm>.py`
  - `tests/prod_env_tests/test_<algorithm>_validation.py`
  - `tests/prod_env_tests/expected/<algorithm>_expected.json`
  - `documentation/algorithms/<algorithm>.md`

## Output Schema

Each scaffold report entry includes:

- `algorithm`
- `phase`
- `check`
- `status`
- `message`
- `path`

Read `references/templates.md` for file placement and template rules.
