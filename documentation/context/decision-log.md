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

## Proposed Decisions

### Decision: One-sample t-test `mean_diff` Reports `mean - mu`

Status: Proposed

Context: `ttest_onesample` returned the sample mean in `mean_diff` and a
confidence interval centred on the sample mean. The UI labels the field "Mean
Difference" and charts it with its interval, and the notebook client plots it
as a mean difference against a reference line at 0.

Decision: `mean_diff` is `mean - mu` and `ci_lower`/`ci_upper` bound
`mean - mu`. The field name is unchanged, so platform-ui and the notebook
client need no change; their existing labels become correct.

Consequences: Consumers that read `mean_diff` as the sample mean get a value
shifted by `mu` (identical when `mu = 0`). Validation fixtures and the R
testcase generator encode the new convention. Rollback is a revert of the
federated core, fixtures, and docs.

Files affected: `exaflow/algorithms/federated/statistics/ttest_onesample.py`,
`tests/*/expected/ttest_onesample_expected.json`,
`tests/testcase_generators/one_sample_testcase_generator.py`,
`documentation/algorithms/TtestOneSample.md`.

Date: 2026-10-13

### Decision: Controller Probes Do Not Depend on Workers

Status: Proposed

Context: The controller startup and liveness probes used `/healthcheck`, which
calls every worker. A single unreachable worker kept the controller failing its
probes and restarting, although the worker landscape aggregator already drops
unreachable workers and serves the remaining federation.

Decision: A partial federation is acceptable for serving. Controller startup
and liveness probes use `/datasets`, which is served from the in-memory
registry. `/healthcheck` keeps its all-workers semantics for CI and manual
checks. Analysis validation no longer assumes a global worker is present.

Consequences: The controller stays up while workers are down; datasets on
unreachable workers are simply absent from `/datasets` until they return.
There is no readiness gate on worker availability.

Files affected: `kubernetes/templates/exaflow-controller.yaml`,
`exaflow/controller/services/worker_landscape_aggregator/worker_landscape_aggregator.py`.

Date: 2026-10-13

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
