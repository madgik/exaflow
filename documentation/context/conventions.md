# Conventions

## Confirmed Conventions

### Python and Formatting

- Python target is 3.10.
- Line length is 88.
- Formatting uses double quotes and spaces.
- Imports are organized as single-line imports.
- Agents must not run Ruff manually; commit hooks and CI own these checks.

### Package Management

- Root Exaflow package uses Poetry.
- `exadata-validator` is a separate Poetry package with its own lockfile.
- CI installs dependencies with Poetry and often uses `--no-root` for root
  Exaflow tests.

### Configuration

- Controller config is loaded in `exaflow/controller/__init__.py`.
- Worker config is loaded and normalized in `exaflow/worker/__init__.py`.
- Aggregation server config is loaded in `exaflow/aggregation_server/__init__.py`.
- Local deployment config starts from `.deployment.sample.toml` and generates
  files under `configs/`.
- Algorithm discovery uses `EXAREME3_ALGORITHM_FOLDERS` and
  `FLOWER_ALGORITHM_FOLDERS`.

### API and DTOs

- Quart endpoint handlers live in `exaflow/controller/quart/endpoints.py`.
- Request DTOs and validators live under `exaflow/controller/services/api`.
- Public analysis form shape is defined by common inputdata, preprocessing, and
  algorithm specifications exposed under `GET /specifications/*`.
- `documentation/api-specification.md` documents the API contract for clients.

### Algorithms

- Runtime Exareme3 algorithm wrappers live under
  `exaflow/algorithms/exareme3`.
- Federated algorithm core logic lives under `exaflow/algorithms/federated`.
- Algorithm specifications are code-defined through `get_specification()`.
- Specification metadata should be UI-ready: compact standard `label`,
  one-sentence method-level `desc`, and detailed parent `documentation` for
  formulas, defaults, ranges, options, assumptions, outputs, and reference
  anchors. Avoid platform/execution terms in user-facing text; prefer neutral
  wording such as "computed from aggregated sufficient statistics without
  sharing raw data" where relevant.
- Input labels should be UI-facing, not raw identifiers such as `x`, `y`,
  `var`, or `vars`. Parameter descriptions should say what the setting does and
  should not repeat schema shape, options, defaults, min/max bounds, or
  requiredness.
- UDF helpers are registered through `@exareme3_udf`.
- New algorithm work should use `.agents/skills/exaflow-algorithm-scaffold` and
  `.agents/skills/exaflow-algorithm-validate`.

### Testing

- Fast tests live under `tests/standalone_tests`.
- Federated algorithm parity tests live under
  `tests/standalone_tests/federated_algorithms`.
- Prod fixtures live under `tests/prod_env_tests/expected`.
- Pytest markers include `slow`, `very_slow`, `smpc`, and `smpc_cluster`.

### Deployment

- Dockerfiles exist for controller, worker, and aggregation server.
- Helm chart lives under `kubernetes`.
- kind/Helm are used by prod environment tests.

## Recommended Conventions To Verify

- Keep business/runtime orchestration out of Quart route handlers where possible;
  delegate to controller services.
- Keep federated math independent of controller/worker runtime so standalone
  tests remain cheap.
- Keep deployment-specific values in config files, Helm values, or environment
  variables rather than hardcoding them in service logic.
- Add regression tests for bug fixes when the behavior can be reproduced in a
  focused standalone test.
- Use prod environment tests only when service wiring, deployment, runtime data
  flow, or public API behavior cannot be validated locally.
- Document public compatibility impact for API, algorithm spec, protobuf, config,
  or Helm value changes.

## Naming

- Algorithm ids should be lower snake_case.
- Algorithm modules, request names, specification names, tests, fixtures, and
  docs should use the same id unless an existing display-style docs convention
  already differs.
- Test modules generally use `test_<thing>.py`.
- Expected fixtures generally use `<algorithm_id>_expected.json`.

## Error Handling

- Use domain-specific exceptions already present in the repo, such as
  `BadInputError`, `BadUserInput`, and validation errors.
- Preserve detailed logs for operators, but do not log secrets or sensitive data.
- Do not hide controller/worker/aggregation failures behind generic success
  responses.

## Dependency Rules

- Do not add production dependencies without explicit justification.
- Keep root and `exadata-validator` dependency changes separate.
- Update lockfiles intentionally when dependency manifests change.
