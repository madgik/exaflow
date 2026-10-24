# Decision Log

Use this file for durable architectural decisions. Mark inferred decisions as
`Inferred / verify` until a maintainer confirms them.

**Do not treat `Inferred / verify` entries as enforced policy.** Prefer
`AGENTS.md`, source code, and `documentation/api-specification.md` when they
conflict with an inferred entry here.

## Template

### Decision: <title>

Status: Proposed / Accepted / Deprecated / Inferred / verify

Context:

Decision:

Consequences:

Files affected:

Date:

## Accepted Decisions

### Decision: Bound Controller Analyses and Keep Liveness Off That Pool

Status: Accepted

Context: The controller is one Hypercorn process. Moving `algorithm.run` off
the event loop let analyses overlap on the default thread pool, which is sized
from node CPUs rather than `max_concurrent_experiments`, and which `/healthcheck`
and `/wla` share. Quart cancels the request task on client disconnect, so an
unshielded `asyncio.to_thread` ran aggregation cleanup while `algorithm.run`
was still using that aggregation context. Startup and liveness probes that call
`/healthcheck`, or that wait `worker_tasks_timeout` seconds, restart a live
controller when a worker is down or the loop is briefly busy.

Decision: Run analysis work, aggregation configure/cleanup, and Flower
start/stop/garbage-collect on a dedicated pool of `max_concurrent_experiments`
threads. Reject extra analyses with HTTP 503. On cancellation, wait for the
in-flight thread before cleanup. Keep `/livez`, `/healthcheck`, and `/wla` off
that pool. Point startup and liveness probes at `/livez` with a 5 second
timeout. Use `/healthcheck` only as a readiness probe.

Consequences: More than `max_concurrent_experiments` simultaneous analyses
receive 503. Client disconnect no longer cleans up aggregation state early.
A down worker removes the controller from Service endpoints and does not
restart it. Worker gRPC pools (`max_concurrent_experiments + 4`) and the
aggregation server pool (`max_concurrent_experiments` per local worker) are
derived from the same limit; the `.deployment.toml` key
`aggregation_server.max_grpc_connections` is gone. There is no separate
overall analysis deadline; per-call worker timeouts still apply. An overall
deadline that returned before the thread stopped would race cleanup again.

Files affected: `exaflow/controller/services/analysis_execution.py`,
`exaflow/controller/services/exareme3/strategies.py`,
`exaflow/controller/services/flower/strategies.py`,
`exaflow/controller/quart/endpoints.py`,
`kubernetes/templates/exaflow-controller.yaml`,
`kubernetes/templates/exaflow-localworker.yaml`, `tasks.py`,
`documentation/api-specification.md`.

Date: 2026-10-24

### Decision: Organize Federated Implementations by Domain

Status: Accepted

Decision: Move outlier reporting and its existing winsorization helpers intact
into `outliers`. Move one-hot and ordinal encoders into `feature_engineering`,
and passthrough into `compose`. Rename mixed-effects `common.py` to
`mixed_effects_utils.py`. Mirror domain placement in tests and update imports,
exports, documentation, and the outlier family mapping in integration tooling.
Shared cross-validation machinery stays in `model_selection/cross_validation`.
This supersedes the earlier choice to retain all federated family locations.

Consequences: Python module paths and family exports change without legacy
aliases. Function and class names, implementations, algorithm IDs,
request/response contracts, numerical behavior, and privacy checks are unchanged.
No dependency, protobuf, Helm-value, or data migration is needed. Deploy
controller and workers from the same revision. Roll back by reverting these
moves and import changes and deploying the previous revision to both services.

Date: 2026-10-19

### Decision: Remove the Transformation-Specific PCA Variant

Status: Accepted

Decision: Remove the separate PCA algorithm accepting per-variable
transformations, its enum entry, integration-tool mappings, validation suite,
fixture, testcase generator, and documentation page. Standard `pca` and its
shared federated core retain their existing behavior.

Consequences: Requests for the removed algorithm are no longer supported and it
is absent from discovery and specifications. No alias or automatic migration is
provided. Standard PCA does not accept the removed transformation parameter.
Deploy controller and workers from the same revision. Rollback requires
restoring the removed variant and deploying that revision to both services.
Privacy thresholds, datasets, dependencies, and deployment settings are unchanged.

Date: 2026-10-19

### Decision: Organize Exareme3 Runtime Modules by Domain

Status: Accepted

Context: A preprocessing folder separated related reports and transformations,
while CV variants were split between model families and model selection.

Decision: Group runtime modules by domain and keep CV wrappers beside their
models. Identify preprocessing by `PreprocessingStep` inheritance, registry
membership, and the specification endpoint. Preserve federated core families.
Keep winsorization validation on `OutlierWinsorizer`, reused by the adjacent
report. Keep K-means replay models and helpers in `kmeans.py`, imported by its
column creator. Direct imports between related runtime modules are appropriate;
sharing alone does not justify extracting another module.
Resolve nested wrappers in integration tools; use `--runtime-domain` when a
runtime domain differs from its federated family. Reject duplicate filenames.

Consequences: Python import paths change without compatibility aliases. UDF
registry keys remain unchanged: they use the module basename and callable name,
which this refactor preserves. Controller and workers should deploy the same
revision. Algorithm/preprocessing IDs, specifications, request/response
shapes, numerical behavior, and privacy checks are unchanged. Rollback requires
reverting the refactor and deploying the previous revision to both services.
No dependency, protobuf, Helm-value, or data migration is involved.

Files affected: Runtime domains and tests, algorithm integration tools and their
skills, setup/author guides, preprocessing and affected algorithm documentation.

Date: 2026-10-19

### Decision: Correct Linear Regression CV Metrics and Name RMSE Explicitly

Status: Accepted

Context: The CV scorer received a default predictor count of zero and replaced
all metrics with zero. The response also named its RMSE field `mean_sq_error`.

Decision: Derive predictor count after encoding, calculate metrics independently,
and replace `mean_sq_error` with `root_mean_sq_error` without an alias. A metric
summary is `[null, null]` if any fold is undefined or infinite. Keep the existing
F diagnostic formula, documented separately from fitted-model inference.

Consequences: API consumers must migrate the field name and accept null
summaries. Controller and workers require a coordinated rollout because the
regression scoring aggregation sequence changes. Rollback must include the
consumer migration. Dataset loading and privacy thresholds are unaffected.

Files affected: The shared CV/scorer modules, linear regression CV runtime,
tests and fixtures, and `documentation/algorithms/linear_regression_cv.md`.

Date: 2026-10-02

## Inferred Decisions

### Decision: Keep Federated Core Logic Separately Testable

Status: Inferred / verify

Context: Runtime wrappers live under `exaflow/algorithms/exareme3`, while core
federated implementations and standalone parity tests live under
`exaflow/algorithms/federated`.

Decision: Algorithm math should remain in federated core modules where possible,
with Exareme3 wrappers handling runtime orchestration and specifications.

Consequences: New algorithm work usually needs both a runtime wrapper and a
standalone-tested federated core.

Files affected: `exaflow/algorithms/exareme3`,
`exaflow/algorithms/federated`, `tests/standalone_tests/federated_algorithms`.

Date: 2026-05-22

### Decision: Use Runtime Specifications As Algorithm Form Source Of Truth

Status: Inferred / verify

Context: `documentation/api-specification.md` states that the
`/specifications/*` endpoints are the runtime source of truth for inputdata,
preprocessing, algorithm availability, and form shape.

Decision: Clients and agents should not hardcode analysis request fields that
are already represented by runtime specifications.

Consequences: Specification changes must be tested and documented as public
contract changes.

Files affected: `exaflow/algorithms/specifications.py`,
`exaflow/controller/services/api`, `documentation/api-specification.md`.

Date: 2026-05-22

### Decision: Use Repository Skills For New Algorithm Work

Status: Inferred / verify

Context: Root `AGENTS.md` and `documentation/new-algorithm-setup.md` identify
repository skills as the canonical new-algorithm workflow.

Decision: Agents should use scaffold and validation skill scripts rather than
manually creating every integration file.

Consequences: New algorithm work is not complete until skill validation passes
without failures or warnings.

Files affected: `.agents/skills`, `documentation/new-algorithm-setup.md`,
algorithm wrapper/core/test/docs paths.

Date: 2026-05-22
