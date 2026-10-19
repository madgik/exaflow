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

## Algorithm Integration Tooling

Runtime wrapper tests follow domain folders under
`tests/standalone_tests/exareme3`; federated core tests retain their family
folders. Test recursive discovery and import idempotence when moving wrappers
or preprocessing steps. Scaffold and validator tools resolve nested wrappers and
reject duplicate filenames; runtime domain and federated family may differ.

```bash
uv run pytest -q tests/standalone_tests/test_algorithm_integration_driver.py tests/standalone_tests/test_import_algorithm_modules_idempotent.py
```

Integration and fast validation run focused tests without Ruff or formatters;
formatting and import-order checks belong to commit hooks and CI.

Federated domain tests follow `outliers`, `feature_engineering`, and `compose`.
Mixed-effects tests remain in `mixed_effects`, exercising the renamed
`mixed_effects_utils.py` through their model imports. During an uncommitted
package move, the algorithm validator can flag a deliberately deleted package
initializer as a missing touched registration file. Check the replacement
package exports and imports directly; do not restore the obsolete package just
to satisfy that check.

The existing-algorithm validator still enforces canonical artifact names. Known
gaps as of the domain refactor: K-means has input-based prod tests without a
`kmeans_expected.json` fixture; regression CV prod tests use
`test_linearregression_cv.py` and `test_logisticregression_cv.py`; logistic CV
standalone coverage lives in `test_classification_cv.py`. These produce four
artifact-layout failures when validating K-means and both regression CV IDs,
despite passing focused tests and runtime import/specification checks. This
refactor does not rename those core tests or manufacture expected fixtures.

## Full Local Standalone Tests

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
