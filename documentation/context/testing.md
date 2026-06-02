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
poetry run pytest -q tests/standalone_tests
```

For algorithm core changes:

```bash
poetry run pytest -q tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py
```

## Full Local Standalone Tests

CI runs:

```bash
poetry run pytest -s -m "not smpc" --cov=exaflow --cov-report=xml:non_smpc_cov.xml tests/standalone_tests --verbosity=4
```

SMPC standalone tests are present but skipped in CI comments because they are
currently failing or require additional resources. Verify before relying on
them.

## Environment-Backed Tests

Algorithm validation tests require local service deployment:

```bash
poetry run inv create-configs
poetry run inv deploy --no-install-dep
poetry run pytest tests/algorithm_validation_tests/exareme3 --verbosity=4
```

Prod environment tests require Docker, kind, Helm, Kubernetes resources, built
images, and generated worker data paths:

```bash
poetry run pytest tests/prod_env_tests --verbosity=4
```

Do not claim these suites passed unless the environment was actually running and
the command completed successfully.

## Nested Validator Tests

Run from the nested package:

```bash
cd exadata-validator
poetry run pytest -q
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
| `exadata-validator` change | `cd exadata-validator && poetry run pytest -q`; build/smoke test if packaging or CLI changes. |
