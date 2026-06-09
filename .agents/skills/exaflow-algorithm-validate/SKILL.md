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
- Ignore import-order and formatting-only lint findings. Automated tools handle
  those mechanical edits.
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

- Fast tier: targeted standalone tests. Ignore import-order and formatting-only
  lint findings; automated tools handle those mechanical edits.
- Strict tier: fast tier + targeted prod_env tests.

Manual specification review:

- `label` values should be compact, standard, user-facing, and not
  implementation-specific.
- Algorithm/preprocessing `desc` values should be compact UI copy: one concise
  method-level sentence suitable for cards/tooltips.
- Detailed formulas/defaults/ranges, parameter option lists, and output
  interpretation belong in the parent algorithm/preprocessing step
  `documentation` field, along with important assumptions and careful reference
  anchors to packages or standard methodology where applicable.
- User-facing spec text should avoid platform/execution terms such as `Exaflow`,
  `exareme`, `MIP`, `worker`, `engine`, `federated implementation`, and
  `aggregation-server-backed`.
- Input labels should be UI-facing and should not be raw identifiers such as
  `x`, `y`, `var`, or `vars`.
- Parameter `desc` values should describe what the setting does, not restate
  schema shape, options, defaults, min/max bounds, requiredness, or dictionary
  structure. Do not add parameter-level `documentation`.
- Omit explicit optional `None` specification arguments such as
  `parameters=None`, `validation=None`, `enumslen=None`, `default=None`,
  `enums=None`, `dict_values_enums=None`, `min=None`, and `max=None`.
- Human-facing algorithm Markdown under `documentation/algorithms/` should have
  a technical structure: overview, inputs, method/model, computation without
  row-level data, aggregated quantities, `### Federated flow` pseudocode,
  technical decisions, outputs, validation reference, and limitations. Output
  fields must match the wrapper Pydantic response model, and materially
  different runtime variants should have separate pages.

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
  - Ignore import-order and formatting-only lint findings. Fix test/runtime
    errors before re-running strict tier.

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

Read `references/path-policy.md` for the validator's path rules and legacy
suite policy.
