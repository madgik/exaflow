---
name: exaflow-algorithm-validate
description: "Validate required Exaflow Exareme3 algorithm implementation steps."
---

# Exaflow Algorithm Validate

Use this skill to verify that Exareme3 algorithm development artifacts and checks are complete.

## Workflow

1. Run validator from the Exaflow repository root.
2. Default selection validates changed algorithms.
3. Use `--algorithms` for explicit targets.
4. Use `--new-algorithm` to enforce full canonical integration checks.
5. Use `--strict` to run standalone + prod_env runtime suites.
6. Inspect JSON output and resolve `failed` entries first, then `warn` entries.

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
- Placeholder token rejection in scaffolded artifacts (`TODO`, `NotImplementedError`).
- Touched registration files are syntactically valid and contain expected symbols.

`--new-algorithm` mode adds canonical integration requirements:

- Federated core exists at `exaflow/algorithms/federated/<family>/<algorithm>.py`.
- Family and root `__init__.py` exports include expected federated symbol.
- `AlgorithmName` enum contains algorithm value.
- Federated README index and federated docs entry exist.
- Canonical file paths are required (legacy compatibility no longer sufficient).
- Expected fixture must be non-empty and contain runnable case shape (`input` + `output`).

Runtime checks:

- Fast tier: `ruff check --select I`, `ruff format --check`, targeted standalone tests.
- Strict tier: fast tier + targeted prod_env tests.

Legacy policy:

- `legacy_used` is a non-failing warning in standard mode.
- `canonical_missing` is failing in `--new-algorithm` mode.

## Mandatory New Federated Algorithm Checklist

1. Scaffold with family-aware options from `$exaflow-algorithm-scaffold`.
2. Implement Exareme3 wrapper and federated core logic.
3. Ensure exports/registrations are patched and importable.
4. Add standalone parity tests.
5. Add prod validation test + non-empty expected fixture.
6. Run validator fast mode.
7. Run validator strict mode when requested.

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
- Canonical missing with legacy present:
  - Create canonical file path while keeping legacy optional.
- Registration symbol missing:
  - Patch `__init__.py`/`AlgorithmName` with expected symbol/value.
- Placeholder check failed:
  - Remove scaffold TODO/NotImplemented placeholders.
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
