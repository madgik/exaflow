---
name: exaflow-algorithm-scaffold
description: "Scaffold and fully integrate new Exaflow Exareme3 algorithms end-to-end. Use when asked to add, create, implement, or integrate a new algorithm (for example Levene's test), including code, tests, docs, and validation."
---

# Exaflow Algorithm Scaffold

Use this skill for end-to-end algorithm integration. Do not stop at placeholders unless the user explicitly asks for placeholder-only output.

Human-facing setup guide: `documentation/new-algorithm-setup.md`.

## Default Execution Policy

1. Scaffold missing files for the target algorithm.
1. Implement algorithm logic in `exaflow/algorithms/exareme3/<algorithm>.py`.
1. Implement federated core in `exaflow/algorithms/federated/<family>/<algorithm>.py`.
1. Implement standalone test coverage in `tests/standalone_tests/federated_algorithms/<family>/test_<algorithm>.py`.
1. Implement prod validation test and expected fixture in `tests/prod_env_tests`.
1. Implement or update docs under `documentation/algorithms/` and `exaflow/algorithms/federated/docs/`.
1. Run validation with `$exaflow-algorithm-validate` (fast, then strict when requested).

## Execution-First Contract

Agents using this skill must keep working until the Definition of Done is met or a concrete blocker is proven by a failing command.

Definition of Done:

- Scaffold command has run successfully.
- All generated placeholder text is replaced (`TODO`, `NotImplementedError`,
  `__REPLACE_ME_*__`).
- Exareme3 wrapper and federated core import successfully.
- `get_specification()` uses UI-ready metadata: compact standard `label`, one
  concise method-level `desc`, and detailed parent algorithm/preprocessing step
  `documentation` for formulas/defaults/ranges, option lists, assumptions,
  parameter mechanics, outputs, and careful reference anchors.
- User-facing spec text avoids platform/execution terms such as `Exaflow`,
  `exareme`, `MIP`, `worker`, `engine`, `federated implementation`, and
  `aggregation-server-backed`; use neutral wording such as "computed from
  aggregated sufficient statistics without sharing raw data" where relevant.
- Input labels are UI-facing, not raw identifiers such as `x`, `y`, `var`, or
  `vars`; prefer labels like `Outcome`, `Covariates`, `Variables`, `Features`,
  `Grouping variable`, or `Additional variables`.
- Parameter `desc` values explain what the setting does, not schema shape. Do
  not repeat options/defaults/min/max/requiredness. Do not add parameter-level
  `documentation`; put parameter details in the parent `documentation`.
- Parameter specs use typed validation (`dict_keys_enums`, `dict_values_enums`,
  `dict_values_type`) instead of duplicating shape/type checks in algorithm
  runtime code.
- Omit explicit optional `None` specification arguments such as
  `parameters=None`, `validation=None`, `enumslen=None`, `default=None`,
  `enums=None`, `dict_values_enums=None`, `min=None`, and `max=None`.
- Standalone parity test exists and passes.
- Prod validation test and non-empty expected fixture exist.
- Registration touchpoints and docs are patched.
- `validate_algorithms.py --new-algorithm <name>` passes with no warnings.
- Focused standalone `pytest` passes. Ignore import order, formatting-only lint,
  and other purely mechanical style issues; automated tools handle them.

Checkpoint responses are only acceptable when they include a failing command, its error summary, and the next concrete fix.

## Watchdog Policy

If a delegated agent has no meaningful file diff or command result after 5-8 minutes, interrupt it and run the integration driver below. After two stalled cycles, stop delegating and finish locally from the driver report.

## Commands

Mechanical integration gate runner:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm my_new_algorithm --family statistics
```

After implementation edits are already present:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm my_new_algorithm --family statistics --skip-scaffold
```

Scaffold one new algorithm with family-aware integration patches:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_new_algorithm --family statistics
```

Plan-only / impact preview:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_new_algorithm --family statistics --dry-run
```

Scaffold explicit standalone subfolder (overrides family inference):

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_new_algorithm --subfolder linear_model
```

## CLI Contract

- Target selection:
  - `--algorithms` for explicit target names.
  - `--all` for runtime-catalog targets.
  - One of the above is required.
- New options:
  - `--family`, `--subfolder`.
  - `--with-federated-core/--no-with-federated-core`.
  - `--with-registration/--no-with-registration`.
  - `--with-doc-index/--no-with-doc-index`.
  - `--with-sample-fixture/--no-with-sample-fixture`.
- Deterministic standalone placement:
  - `--subfolder` if provided.
  - Else family-derived subfolder.
  - Else `_generated` with warning in JSON report.
- Overwrite policy: never overwrite existing files.

## Mandatory New Federated Algorithm Checklist

1. Run scaffold with `--algorithms <name>` and `--family <family>`.
1. Implement Exareme3 wrapper logic in `exaflow/algorithms/exareme3/<name>.py`.
1. Write `get_specification()` text for the UI: use compact standard labels,
   one concise method-level `desc`, and parent `documentation` for formulas,
   defaults, ranges, strategy options, assumptions, outputs, result
   interpretation, and reference anchors. Parameter `desc` values should say
   what the setting does rather than restating schema shape, options, defaults,
   min/max bounds, or requiredness. Avoid advisory "when to use" language unless
   it is required product copy.
1. Implement federated core in `exaflow/algorithms/federated/<family>/<name>.py`.
1. Confirm registration patches:
   - `exaflow/algorithms/federated/<family>/__init__.py`
   - `exaflow/algorithms/federated/__init__.py`
   - `exaflow/algorithms/specifications.py` (`AlgorithmName`)
   - `exaflow/algorithms/federated/README.md`
1. Replace scaffold placeholders in standalone and prod tests.
1. Populate expected fixture with at least one runnable test case.
1. Run validate fast mode, then strict mode when requested.

## Family Cookbook

- `statistics`:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_stat_algo --family statistics
```

- `linear_model`:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_linear_algo --family linear_model
```

- `decomposition`:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_decomp_algo --family decomposition
```

- `naive_bayes`:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms my_nb_algo --family naive_bayes
```

## Troubleshooting

- Import failure in generated module:
  - Verify `exaflow/algorithms/exareme3/<algorithm>.py` imports and class names.
- Subfolder fallback warning:
  - Re-run with explicit `--family` or `--subfolder`.
- Registration missing symbol:
  - Re-run scaffold with `--with-registration`.
- Ruff format/import failures:
  - Ignore import order and formatting-only lint. Automated tools handle these
    mechanical edits.
- Strict mode fails in prod tests:
  - Ensure environment prerequisites and fixture values are realistic for available datasets.

## Output Schema

Each scaffold report entry includes:

- `algorithm`
- `phase`
- `check`
- `status`
- `severity` (`pass|warn|failed`)
- `message`
- `next_action`
- `path`

Read `references/templates.md` for file placement and template rules.
