# Testing

Testing strategy and validation matrix for Exaflow. For agent rules and focused
test commands, start with `AGENTS.md` at the repository root.

## Frameworks Detected

- Root test framework: pytest.
- Root plugins/dependencies: pytest-xdist, pytest-subtests, pytest-cov.
- Nested validator test framework: pytest.
- CI coverage upload: Qlty coverage action.

## Test Directory Structure

- `tests/standalone_tests`: local tests that should not need a full Exaflow
  environment.
- `tests/standalone_tests/federated_algorithms`: fast parity tests for
  federated algorithm cores.
- `tests/algorithm_validation_tests`: environment-backed algorithm validation
  tests using local services from `inv deploy`.
- `tests/prod_env_tests`: production-like tests using Docker, kind, Helm, and
  Exaflow service pods.
- `tests/prod_env_tests/expected`: expected algorithm result fixtures.
- `tests/test_data`: CSV and CDE metadata fixtures.
- `tests/testcase_generators`: helpers for generating expected cases; excluded
  from pytest recursion in `pyproject.toml`.
- `exadata-validator/tests`: tests for the nested validator package.

## Fast Tests

Use focused standalone tests while developing:

```bash
uv run pytest -q tests/standalone_tests
```

For algorithm core changes:

```bash
uv run pytest -q tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py
```

## Full Local Standalone Tests

FETA's focused suite includes simulated multi-worker execution with an
independent mock aggregation server, plus the real Quart application, DuckDB
loading, and localhost gRPC workers/aggregation services. Worker membership and
metadata are provided by a test landscape; this is package integration, not a
Kubernetes deployment test:

```bash
uv run pytest -q tests/standalone_tests/feta
```

CI runs:

```bash
uv run pytest -s -m "not smpc" --cov=exaflow --cov-report=xml:non_smpc_cov.xml tests/standalone_tests --verbosity=4
```

SMPC standalone tests are present but skipped in CI comments because they are
currently failing or require additional resources. Verify before relying on
them.

## Environment-Backed Tests

Algorithm validation tests require local service deployment:

```bash
uv run inv create-configs
uv run inv deploy --no-install-dep
uv run pytest tests/algorithm_validation_tests/exareme3 --verbosity=4
```

Prod environment tests require Docker, kind, Helm, Kubernetes resources, built
images, and generated worker data paths:

```bash
uv run pytest tests/prod_env_tests --verbosity=4
```

Do not claim these suites passed unless the environment was actually running and
the command completed successfully.

## Nested Validator Tests

Run from the nested package:

```bash
cd exadata-validator
uv run pytest -q
```

CI also builds the package and smoke-tests the wheel.

## Fixtures and Mocks

- Data-model fixtures live under `tests/test_data`.
- Prod expected result fixtures live under `tests/prod_env_tests/expected`.
- Algorithm validation expected fixtures live under
  `tests/algorithm_validation_tests/exareme3/expected`.
- Worker data path generation is handled by `tests/worker_data_paths_builder.py`.
- Federated algorithm test helpers live under
  `tests/standalone_tests/federated_algorithms/utils`.

## Critical Paths That Need Strong Coverage

- Algorithm request validation and specification DTO compatibility.
- Dynamic algorithm discovery and UDF registration.
- Worker data loading, preprocessing, and minimum-row privacy checks.
- Aggregation server configure/aggregate/cleanup lifecycle.
- gRPC protobuf compatibility between controller, workers, and aggregation
  server.
- Deployment templates that affect service discovery, config, volumes, or image
  tags.
- `exadata-validator` CSV/CDE validation behavior and CLI reporting.

## Validation Matrix

| Change type | Required validation |
| --- | --- |
| Docs-only change | `git diff --check`; inspect rendered paths/links where practical. |
| Algorithm core change | Focused `tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py`; algorithm validator when relevant. |
| Algorithm wrapper/spec change | Focused standalone tests plus spec/request validation tests or algorithm validation skill. |
| API behavior change | Controller standalone API tests and relevant prod/algorithm validation tests if runtime behavior changes. |
| Worker data/UDF change | Focused worker/UDF standalone tests; environment-backed tests if gRPC/runtime behavior changes. |
| Aggregation change | `tests/standalone_tests/aggregation_server`; focused algorithm tests that use aggregation server. |
| Config/deployment change | `helm template kubernetes/`; relevant Invoke dry/manual checks; prod environment tests when feasible. |
| Protobuf change | Regenerate generated files; run gRPC/aggregation/controller tests. |
| Privacy/SMPC/DP change | Focused tests plus human review; run environment-backed tests if behavior changes. |
| Dependency update | Lockfile update, focused tests, lint/format, and CI-equivalent suite where feasible. |
| Refactor without behavior change | Focused tests for touched modules and broader standalone tests if shared code moved. |
| `exadata-validator` change | `cd exadata-validator && uv run pytest -q`; build/smoke test if packaging or CLI changes. |

FETA deployment parity check (requires the deployed controller at port 5100):

```bash
uv run pytest -q tests/prod_env_tests/test_feta_pearson_correlation.py
```

This compares FETA and existing Pearson on identical complete-case selections,
including omitted `x`. Structure and counts are exact; numerical values allow
only summation-order differences (`rtol=1e-10`, `atol=1e-12`).

One-sample t-test deployment parity, after development federation readiness:

```bash
uv run pytest -q tests/prod_env_tests/test_feta_ttest_onesample.py
```

The test selects complete cases identically for both algorithms, retains the
normal algorithm's required missing-value handler, and sends no preprocessing
to FETA. It compares every response field, including one-sided infinity markers.

Paired t-test deployment parity, after development federation readiness:

```bash
uv run pytest -q tests/prod_env_tests/test_feta_ttest_paired.py
```

Both implementations receive identical complete-case filters on x and y.
Every output field is compared, retaining x-minus-y direction and one-sided
infinity markers. FETA requests contain no preprocessing.

Independent t-test FETA parity is covered by
`tests/standalone_tests/feta/test_feta_ttest_independent.py` and real package tests
in `tests/standalone_tests/feta/test_package_integration.py`. After development
`inv deploy` and dataset discovery, run
`uv run pytest -q tests/prod_env_tests/test_feta_ttest_independent.py` to compare
all original fixture cases against normal independent t-test through `/analysis`.
Both requests use identical complete-case filters; FETA has no preprocessing.

One-way ANOVA FETA parity is covered by
`tests/standalone_tests/feta/test_feta_anova_oneway.py`, with aggregation-contract
checks in `test_extrema_and_unique.py` and real package `/analysis` integration.
After development deployment and dataset discovery, run
`uv run pytest -q tests/prod_env_tests/test_feta_anova_oneway.py` to compare all
five original cases and every nested response field against normal ANOVA.

`tests/standalone_tests/feta/test_dependency_boundary.py` enforces production FETA
ownership: no Exareme3, shared federated-algorithm, data-helper/filter or numeric
serialization implementation imports. Named public platform contracts are allowed;
legacy algorithm imports are confined to test reference paths. Package integration
uses FETA's owned data loader, and live parity checks cover all five algorithms.

Two-way ANOVA FETA coverage is in
`tests/standalone_tests/feta/test_feta_anova_twoway.py` and `test_design_operations.py`,
plus real `/analysis` package tests for both sums-of-squares types. Development
parity uses `uv run pytest -q tests/prod_env_tests/test_feta_anova_twoway.py` after
`inv deploy` and dataset discovery. Tests compare the complete normal response
on identical complete-case selections; full-rank simulations also use statsmodels.
