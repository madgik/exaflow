---
name: exaflow-algorithm-validate
description: "Validate required Exaflow Exareme3 algorithm implementation steps."
---

# Exaflow Algorithm Validate

Use this skill to verify that Exareme3 algorithm development artifacts and checks are complete.

Human-facing setup guide: `documentation/new-algorithm-setup.md`.

## Workflow

1. Run validator from the Exaflow repository root.
1. Default selection validates changed algorithms.
1. Use `--algorithms` for explicit targets.
1. Use `--new-algorithm` to enforce full canonical integration checks.
1. Use `--strict` to run standalone + prod_env runtime suites.
1. Inspect JSON output and resolve `failed` entries first, then `warn` entries.

## Execution-First Contract

Agents using this skill must treat validator output as the source of truth. A task is not complete while `failed` or `warnings` are non-empty.

Definition of Done:

- `validate_algorithms.py --new-algorithm <name>` exits successfully.
- Validator JSON contains no `failed` entries.
- Validator JSON contains no `warnings` entries.
- Focused standalone tests pass.
- Focused import-order and format checks pass.
- Strict mode is run when the user requests prod-env confidence.

If validation fails, continue fixing until the same command passes. If blocked, return only the failing command, the relevant error, files already changed, and the next required action.

## Watchdog Policy

When supervising a delegated validation run, request a status snapshot after 5-8 minutes with commands run, files changed, and current failing gate. If the agent cannot name a current failing gate, interrupt it and run the integration driver from the scaffold skill.

## Commands

Validate changed algorithms (default selection):

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root .
```

Validate explicit targets:

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --algorithms my_algorithm
```

Validate all runtime-catalog algorithms:

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --all
```

New-algorithm strict checklist:

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm my_new_algorithm --strict
```

## Validation Contract

Static checks (always):

- Runtime catalog membership.
- Algorithm module exists/imports and `get_specification().name` matches id.
- Required files exist (standalone, prod test, expected fixture, docs).
- Placeholder token rejection in scaffolded artifacts (`TODO`,
  `NotImplementedError`, `__REPLACE_ME_*__`).
- Touched registration files are syntactically valid and contain expected symbols.

`--new-algorithm` mode adds canonical integration requirements:

- Federated core exists at `exaflow/algorithms/federated/<family>/<algorithm>.py`.
- Family and root `__init__.py` exports include expected federated symbol.
- `AlgorithmName` enum contains algorithm value.
- Federated README index and federated docs entry exist.
- Expected fixture must be non-empty and contain runnable case shape (`input` + `output`).

Runtime checks:

- Fast tier: `ruff check --select I`, `ruff format --check`, targeted standalone tests.
- Strict tier: fast tier + targeted prod_env tests.

Path policy:

- Required implementation, test, fixture, and documentation paths are enforced directly.
- There is no legacy fallback for alternate prod test, expected fixture, or standalone test filenames.

## Mandatory New Federated Algorithm Checklist

1. Scaffold with family-aware options from `$exaflow-algorithm-scaffold`.
1. Implement Exareme3 wrapper and federated core logic.
1. Ensure exports/registrations are patched and importable.
1. Add standalone parity tests.
1. Add prod environment test + non-empty expected fixture.
1. Run validator fast mode.
1. Run validator strict mode when requested.

## Family Cookbook

- Statistics-family new algorithm:

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm my_stat_algo
```

- Linear-model family new algorithm:

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm my_linear_algo
```

## Troubleshooting

- Runtime catalog membership failure:
  - Verify module discovery and `get_specification().name`.
- Required path missing:
  - Create the exact path reported by the validator.
- Registration symbol missing:
  - Patch `__init__.py`/`AlgorithmName` with expected symbol/value.
- Placeholder check failed:
  - Remove scaffold `TODO`, `NotImplementedError`, and `__REPLACE_ME_*__` placeholders.
- Ruff/runtime failures:
  - Fix lint/test errors before re-running strict tier.

## Output Schema

Each validator report entry includes:

- `algorithm`
- `phase`
- `check`
- `status`
- `severity` (`pass|warn|failed`)
- `message`
- `next_action`
- `path`

Read `references/compatibility-map.md` for transition mappings.
