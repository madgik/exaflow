# New Algorithm Setup

This is the fastest supported path for adding a new Exareme3 federated
algorithm to Exaflow. It is written for algorithm developers and Codex agents
working from the repository root.

## What You Need First

- Python dependencies installed with `poetry install`.
- A lower snake_case algorithm id, for example `ttest_welch`.
- A federated family, for example `statistics`, `linear_model`,
  `decomposition`, or `naive_bayes`.
- A real statistical contract: input variables, parameters, output fields, and
  at least one expected test case.

Keep the same algorithm id everywhere: module name, specification name,
runtime request name, test names, fixture name, and documentation name.

## One Prompt For Codex

Use this prompt when you want Codex to integrate a new algorithm end-to-end:

```text
Use the exaflow-algorithm-scaffold and exaflow-algorithm-validate skills to
integrate <algorithm_id> as a new federated <family> algorithm in this Exaflow
repo.

Do not stop at placeholders. Implement the Exareme3 wrapper, federated core,
registrations, standalone parity tests, prod validation fixture/test, and
algorithm docs.

Start with:
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family>

After implementation edits, keep rerunning:
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold

Keep fixing until the integration driver reports "done": true. Use --strict
only when the prod environment is available. Return the changed files, commands
run, elapsed time, and any blocker.
```

Example for a statistics algorithm:

```bash
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm ttest_welch --family statistics
```

Concrete prompt example:

```text
Use the exaflow-algorithm-scaffold and exaflow-algorithm-validate skills to
integrate ttest_welch as a new federated statistics algorithm in this Exaflow
repo.

Do not stop at placeholders. Implement the Exareme3 wrapper, federated core,
registrations, standalone parity tests, prod validation fixture/test, and
algorithm docs.

Start with:
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm ttest_welch --family statistics

After implementation edits, keep rerunning:
poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm ttest_welch --family statistics --skip-scaffold

Keep fixing until the integration driver reports "done": true. Use --strict
only when the prod environment is available. Return the changed files, commands
run, elapsed time, and any blocker.
```

## Human Workflow

1. Preview the files and integration patches:

   ```bash
   poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms <algorithm_id> --family <family> --dry-run
   ```

1. Scaffold the algorithm:

   ```bash
   poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/scaffold_algorithms.py --repo-root . --algorithms <algorithm_id> --family <family>
   ```

1. Replace the generated placeholders with real implementation and tests.

1. Run the mechanical integration gate:

   ```bash
   poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold
   ```

1. If the prod environment is available, run the strict gate:

   ```bash
   poetry run python .agents/skills/exaflow-algorithm-scaffold/scripts/integrate_new_algorithm.py --repo-root . --algorithm <algorithm_id> --family <family> --skip-scaffold --strict
   ```

The gate fails on warnings for new algorithms. Treat every `failed` and `warn`
entry as something to fix before the integration is complete.

## Files You Normally Implement

| File | Purpose |
| --- | --- |
| `exaflow/algorithms/exareme3/<algorithm_id>.py` | Exareme3 wrapper, specification, request parsing, worker UDF orchestration, response model. |
| `exaflow/algorithms/federated/<family>/<algorithm_id>.py` | Federated/statistical core logic that can be tested without the full runtime. |
| `tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py` | Fast parity tests for the federated core and edge cases. |
| `tests/prod_env_tests/test_<algorithm_id>_validation.py` | Prod validation test wired to Exaflow request fixtures. |
| `tests/prod_env_tests/expected/<algorithm_id>_expected.json` | Non-empty expected fixture with at least one runnable case. |
| `documentation/algorithms/<algorithm_id>.md` | User-facing algorithm description and request shape. |
| `exaflow/algorithms/federated/docs/<algorithm_id>.md` | Federated implementation notes. |

The scaffold also patches common integration touchpoints when enabled:

- `exaflow/algorithms/federated/<family>/__init__.py`
- `exaflow/algorithms/federated/__init__.py`
- `exaflow/algorithms/specifications.py` (`AlgorithmName`)
- `exaflow/algorithms/federated/README.md`

If the validator reports a missing symbol in one of those files, either rerun
the scaffold with registration enabled or patch the reported file manually.

## Family Defaults

| Family | Use when | Command option |
| --- | --- | --- |
| Statistics | Tests, descriptive statistics, correlations, distribution checks. | `--family statistics` |
| Linear model | Linear/logistic model style algorithms and regressions. | `--family linear_model` |
| Decomposition | PCA/SVD-like algorithms. | `--family decomposition` |
| Naive Bayes | Naive Bayes classifiers and related probability models. | `--family naive_bayes` |

If the standalone test path should not match the family name, pass
`--subfolder <folder>` to both scaffold and integration commands.

## Validation Commands

Direct new-algorithm validation:

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm <algorithm_id>
```

Strict validation with prod tests:

```bash
poetry run python .agents/skills/exaflow-algorithm-validate/scripts/validate_algorithms.py --repo-root . --new-algorithm <algorithm_id> --strict
```

Focused standalone test:

```bash
poetry run pytest -q tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py
```

Focused formatting gates:

```bash
poetry run ruff check --select I exaflow/algorithms/exareme3/<algorithm_id>.py exaflow/algorithms/federated/<family>/<algorithm_id>.py tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py tests/prod_env_tests/test_<algorithm_id>_validation.py
poetry run ruff format --check exaflow/algorithms/exareme3/<algorithm_id>.py exaflow/algorithms/federated/<family>/<algorithm_id>.py tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py tests/prod_env_tests/test_<algorithm_id>_validation.py
```

## Definition Of Done

- No scaffold placeholders remain in generated target files (`TODO`,
  `NotImplementedError`, `__REPLACE_ME_*__`).
- `get_specification().name` matches the algorithm id.
- The algorithm appears in the runtime catalog.
- Exareme3 wrapper and federated core import successfully.
- Standalone parity tests pass.
- Prod validation test and expected fixture exist, and the fixture has at least
  one runnable test case.
- User docs and federated docs exist.
- New-algorithm validation passes with no failed rows and no warnings.
- The integration driver reports `"done": true`.

## Common Failures

| Failure | Fix |
| --- | --- |
| Subfolder fallback warning | Re-run with `--family <family>` or explicit `--subfolder <folder>`. |
| Runtime catalog membership failure | Check `get_specification().name`, `AlgorithmName`, and module discovery under `exaflow/algorithms/exareme3`. |
| Registration symbol missing | Patch the reported `__init__.py` or rerun scaffold with `--with-registration`. |
| Placeholder check failed | Replace generated `TODO`, `NotImplementedError`, and `__REPLACE_ME_*__` values. |
| Ruff import or format failure | Run `poetry run ruff check --select I --fix <files>` and `poetry run ruff format <files>`. |
| Strict prod test failure | Confirm the prod environment is running and expected fixture values match available datasets. |
