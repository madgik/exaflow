---
name: exaflow-algorithm-validate
description: "Validate required Exaflow Exareme3 algorithm implementation steps."
---

# Exaflow Algorithm Validate

Use this skill to verify that Exareme3 algorithm development artifacts and checks are complete.

## Workflow

1. Run the validator from the Exaflow repository root.
2. Default mode validates changed algorithms only.
3. Use `--algorithms` for explicit targets.
4. Use `--strict` to run both standalone and prod_env runtime suites.
5. Inspect the JSON report and fix `failed` entries.

## Commands

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root .
```

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --algorithms my_new_algorithm --strict
```

## Validation Contract

Static checks (always):

- Algorithm module exists and imports.
- `get_specification().name` matches algorithm id.
- Standalone test exists under `tests/standalone_tests/federated_algorithms` (including `_generated` fallback).
- Prod test exists (`test_<algorithm>_validation.py` or compatibility mapping).
- Prod expected fixture exists (`<algorithm>_expected.json` or compatibility mapping).
- Doc exists (`documentation/algorithms/<algorithm>.md` or compatibility mapping).

Runtime checks:

- Fast tier: `ruff check --select I`, `ruff format --check`, targeted standalone tests.
- Strict tier: fast tier + targeted prod_env tests.

Legacy policy:

- `tests/algorithm_validation_tests` is informational only and never required to pass.

Read `references/compatibility-map.md` for transition mappings.
