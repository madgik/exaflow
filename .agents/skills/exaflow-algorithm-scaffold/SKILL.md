---
name: exaflow-algorithm-scaffold
description: "Scaffold and fully integrate new Exaflow Exareme3 algorithms end-to-end. Use when asked to add, create, implement, or integrate a new algorithm (for example Levene's test), including code, tests, docs, and validation."
---

# Exaflow Algorithm Scaffold

Use this skill for end-to-end algorithm integration. Do not stop at placeholders unless the user explicitly asks for placeholder-only output.

## Default Execution Policy

1. Scaffold missing files for the target algorithm.
2. Implement algorithm logic in `exaflow/algorithms/exareme3/<algorithm>.py`.
3. Implement federated core in `exaflow/algorithms/federated/<family>/<algorithm>.py`.
4. Implement standalone test coverage in `tests/standalone_tests/federated_algorithms/<family>/test_<algorithm>.py`.
5. Implement prod validation test and expected fixture in `tests/prod_env_tests`.
6. Implement or update docs under `documentation/algorithms/` and `exaflow/algorithms/federated/docs/`.
7. Run validation with `$exaflow-algorithm-validate` (fast, then strict when requested).

## Commands

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
2. Implement Exareme3 wrapper logic in `exaflow/algorithms/exareme3/<name>.py`.
3. Implement federated core in `exaflow/algorithms/federated/<family>/<name>.py`.
4. Confirm registration patches:
   - `exaflow/algorithms/federated/<family>/__init__.py`
   - `exaflow/algorithms/federated/__init__.py`
   - `exaflow/algorithms/specifications.py` (`AlgorithmName`)
   - `exaflow/algorithms/federated/README.md`
5. Replace scaffold placeholders in standalone and prod tests.
6. Populate expected fixture with at least one runnable test case.
7. Run validate fast mode, then strict mode when requested.

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
  - Run `poetry run ruff check --select I <files>` and `poetry run ruff format <files>`.
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
