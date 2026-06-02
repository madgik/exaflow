# Module Index

## `exaflow/controller`

Purpose: HTTP API, runtime config, worker discovery, execution strategies, and
controller-side task orchestration.

Key files: `quart/endpoints.py`, `services/api/*`,
`services/exareme3/strategies.py`, `services/exareme3/algorithm_flow_engine_interface.py`.

Used by: `run_algorithm`, prod tests, algorithm validation tests, workers, and
deployment manifests.

Rules: Keep route parsing in Quart, validation in API service modules, and
execution orchestration in strategy/task-handler modules.

Tests: `tests/standalone_tests/controller`, `tests/prod_env_tests`.

Notes: Request/spec DTOs are part of the public API contract.

## `exaflow/worker`

Purpose: gRPC worker service, DuckDB dataset loading, worker metadata, and UDF
execution.

Key files: `grpc_server.py`, `exareme3/udf/udf_service.py`,
`worker_info/*`, `utils/duck_db_csv_loader.py`.

Used by: Controller task handlers, local deployment, prod environment tests.

Rules: Preserve privacy minimum-row checks and config loading through
`exaflow/worker/__init__.py`.

Tests: `tests/standalone_tests/test_duck_db_csv_loader_types.py`,
`tests/standalone_tests/exareme3/test_udf_service.py`,
`tests/prod_env_tests`.

Notes: Worker startup loads CSV data into DuckDB.

## `exaflow/aggregation_server`

Purpose: Optional gRPC microservice for federated vector aggregation.

Key files: `server.py`, `constants.py`, `config.toml`,
`exaflow/protos/aggregation_server/*`.

Used by: Controller aggregation client and worker UDF aggregation client.

Rules: Preserve request lifecycle: configure, aggregate, unregister, cleanup.

Tests: `tests/standalone_tests/aggregation_server`.

Notes: `UNION` returns JSON-encoded bytes in the tensor field.

## `exaflow/algorithms/exareme3`

Purpose: Runtime algorithm wrappers, preprocessing steps, in-code
specifications, and UDF registrations.

Key files: family folders, `utils/algorithm.py`,
`utils/preprocessing_step.py`, `utils/registry.py`.

Used by: Dynamic discovery in `exaflow/__init__.py`, controller strategies, and
worker UDF services.

Rules: Keep `get_specification().name` aligned with algorithm ids. Register UDFs
with `@exareme3_udf`.

Tests: Standalone algorithm tests, prod validation tests, integration skill
validators.

Notes: Discovery is controlled by `EXAREME3_ALGORITHM_FOLDERS`.

## `exaflow/algorithms/federated`

Purpose: Testable federated/statistical core logic and implementation docs.

Key files: family folders such as `statistics`, `linear_model`,
`decomposition`, `naive_bayes`, `mixed_effects`, plus `docs/`.

Used by: Exareme3 wrappers and standalone parity tests.

Rules: Keep core logic independent of full controller/worker runtime when
possible.

Tests: `tests/standalone_tests/federated_algorithms`.

Notes: Prefer extending existing family utilities over duplicating math.

## `exaflow/protos`

Purpose: gRPC `.proto` definitions and generated Python modules.

Key files: `worker/worker.proto`, `aggregation_server/aggregation_server.proto`.

Used by: Controller clients, worker service, aggregation server.

Rules: Proto changes are public interface changes and require regenerated files
and compatibility review.

Tests: gRPC and aggregation standalone tests.

Notes: Generated files are expected to stay consistent with source `.proto`.

## `tests`

Purpose: Standalone, environment-backed, prod, data, and generator tests.

Key files: `README.md`, `standalone_tests`, `algorithm_validation_tests`,
`prod_env_tests`, `test_data`, `worker_data_paths_builder.py`.

Used by: CI and local validation.

Rules: Run the most focused suite first. Prod tests require Docker/kind/Helm.

Tests: This directory is the test surface.

Notes: Expected prod fixtures live under `tests/prod_env_tests/expected`.

## `documentation`

Purpose: API docs, algorithm docs, setup guides, user stories, and context
documentation.

Key files: `api-specification.md`, `new-algorithm-setup.md`, `algorithms/*`,
`context/*`.

Used by: Developers, agents, UI/client implementers, and reviewers.

Rules: Keep docs evidence-based. Mark unknowns as `Unknown / TODO: verify`.

Tests: Markdown formatting can be checked through pre-commit/mdformat when
available.

Notes: `documentation/context` is the canonical context layer.

## `kubernetes`

Purpose: Helm chart, values, templates, and Kubernetes deployment docs.

Key files: `Chart.yaml`, `values.yaml`, `templates/*`,
`DevDeployment.md`, `docs/ExaflowDeployment.md`.

Used by: Prod environment tests and deployment workflows.

Rules: Treat image, service, config, volume, and secret changes as high review.

Tests: `helm template kubernetes/`, prod environment workflow.

Notes: kind is used for CI-like prod environment validation.

## `exadata-validator`

Purpose: Separate Poetry package that validates data-model folders using
DuckDB.

Key files: `pyproject.toml`, `validator/commands.py`,
`validator/duckdb_validator.py`, `tests/*`.

Used by: CLI users and package publishing workflow.

Rules: Run commands from `exadata-validator/`; keep package metadata separate
from the root package.

Tests: `cd exadata-validator && poetry run pytest -q`.

Notes: Package exposes `exadata-validator` and `python -m validator`.

## `.agents/skills`

Purpose: Repository-specific Codex skills for algorithm scaffolding and
validation.

Key files: `exaflow-algorithm-scaffold/SKILL.md`,
`exaflow-algorithm-validate/SKILL.md`, skill scripts.

Used by: Agents and developers adding or validating algorithms.

Rules: Use skills for new algorithm work instead of manual file creation.

Tests: Skill scripts validate expected files, docs, fixtures, exports, and
runtime discovery.

Notes: New algorithm work is incomplete until validator warnings/failures are
resolved.

## `tasks.py`

Purpose: Invoke tasks for local deployment, generated configs, data loading,
service lifecycle, SMPC helpers, and cleanup.

Key files: This file.

Used by: README setup flow, CI algorithm validation, local development.

Rules: Treat cleanup and deployment tasks as potentially destructive. Verify
environment paths before changing them.

Tests: Covered indirectly by deployment-backed tests.

Notes: Some tasks write under `configs/`, `tests/test_data/.data_paths`, and
`/tmp`.

## `run_algorithm`

Purpose: CLI helper for building algorithm request payloads and posting to the
controller.

Key files: This script.

Used by: Developers debugging algorithms and expected fixtures.

Rules: Preserve dry-run behavior and JSON merge semantics.

Tests: No dedicated test found during inspection; validate manually if changed.

Notes: Can accept piped JSON and override fields with CLI args.
