______________________________________________________________________

## name: exaflow-algorithm-scaffold description: Create missing placeholder artifacts for Exaflow Exareme3 algorithm development using a shared template. Use when you need to scaffold one or more algorithms, especially for bulk placeholder generation across the runtime algorithm catalog.

# Exaflow Algorithm Scaffold

Use this skill to generate missing placeholder artifacts for Exareme3 algorithms without overwriting existing files.

## Workflow

1. Run the scaffold CLI from the Exaflow repository root.
1. Use `--algorithms` for explicit targets or omit it to scaffold all runtime catalog algorithms.
1. Use `--dry-run` first when checking impact.
1. Review the JSON report (`created`, `skipped_existing`, `failed`).

## Commands

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --dry-run
```

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_new_algorithm
```

## Behavior Contract

- Source of truth for `--all`: runtime Exareme3 algorithm catalog from `exaflow.exareme3_algorithm_classes`.
- `--algorithms` accepts new algorithm IDs that are not yet in the runtime catalog.
- Overwrite policy: never overwrite existing files.
- Standalone test placement: infer `tests/standalone_tests/federated_algorithms/<subfolder>/test_<algorithm>.py`; fallback to `_generated` when uncertain.
- Created placeholders:
  - `exaflow/algorithms/exareme3/<algorithm>.py`
  - `tests/standalone_tests/federated_algorithms/<subfolder>/test_<algorithm>.py`
  - `tests/prod_env_tests/test_<algorithm>_validation.py`
  - `tests/prod_env_tests/expected/<algorithm>_expected.json`
  - `documentation/algorithms/<algorithm>.md`

## Output Schema

Each report entry includes:

- `algorithm`
- `phase`
- `check`
- `status`
- `message`
- `path`

Read `references/templates.md` for template and placement details.
