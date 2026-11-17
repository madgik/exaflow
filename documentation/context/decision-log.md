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

### Decision: CDE Metadata Carries a Single Variable Type

Status: Accepted

Context: The MIP variable-type refactor drops `sql_type` and `isCategorical`
from `CDEsMetadata.json`; `type` alone carries the classification.

Decision: CDEs carry `type` in `continuous`, `discrete`, `nominal`, `ordinal`,
`binary` (`exaflow.datatypes.CDEType`). Storage and categorical status are
derived from it: `continuous` is DOUBLE, `discrete` is INTEGER, categorical types
are integer codes (`CATEGORICAL_CDE_TYPES`; see the later integer-code decision). Metadata dicts, the worker proto,
`GET /cdes_metadata`, and preprocessing output columns use `type` only.
Algorithm input specifications keep `types`/`stattypes`; the controller maps CDE
types onto them. Categorical CDEs without enumerations (former free-text `text`
columns) are rejected as algorithm inputs. `exadata-validator` rejects
numerical CDEs with enumerations, categorical CDEs with `minValue`/`maxValue`,
fractional `discrete` bounds, and `binary` CDEs without exactly two
enumerations.

Consequences: Not backwards compatible. Old data models must be migrated
(`real` → `continuous`, `integer` → `discrete`, `nominal`/`text` → `nominal`,
drop `sql_type`/`isCategorical`, drop empty enumerations). Controller and
workers must be upgraded together. Platform backend/UI consumers of
`sql_type`/`is_categorical` must switch to `type`.

Files affected: `exaflow/datatypes.py`, `exaflow/worker_communication.py`,
`exaflow/protos/worker/worker.proto`, `exaflow/worker/utils/duck_db_csv_loader.py`,
`exaflow/controller/services/api/analysis_request_validator.py`,
`exaflow/algorithms/specifications.py`, `exadata-validator/validator/dataelements.py`.

Date: 2026-10-26

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

### Decision: Match Binary and Ordinal Inputs to CDE Types Exactly

Status: Accepted

Context: With the CDE variable-type refactor, binary and ordinal variables are
declared in the data model, but algorithm slots only distinguished numerical
from nominal. Algorithms that need a binary or ordinal variable accepted any
nominal variable and converted it (`positive_class` collapsing a multi-level
nominal into 0/1, a user supplied `category_order` ranking a nominal variable).

Decision: `InputDataStatType` gains `binary` and `ordinal`. A slot matches CDE
types one-to-one; a nominal CDE never satisfies a `binary` or `ordinal` slot.
Slots that take any categorical variable (covariates, grouping variables,
generic class labels) list `nominal`, `ordinal` and `binary` explicitly.
`logistic_regression`, `logistic_regression_cv`, `glmm_binary` (outcome) and
`fisher_exact` (both variables) require `binary`; `glmm_ordinal` (outcome)
requires `ordinal`. Cox event variables are chosen with a parameter, so
`ParameterSpecification.variable_stattypes` requires `binary` for `event_var`.
The controller validates all of these and the algorithms repeat the check
(`algorithms/exareme3/utils/cde_types.py`). `positive_class` selects one of the
two declared levels and never collapses levels. `glmm_ordinal` takes its level
order from the data model; `category_order` is optional and must equal it.
`FederatedOneHotEncoder` and `FederatedOrdinalEncoder` stay inside the
algorithms as private modules and are not exported.

Consequences: Requests that passed a multi-level nominal variable as the outcome
of these algorithms are now rejected. Specification JSON gains `binary` and
`ordinal` stattypes and `variable_stattypes`; clients that filter variables by
`stattypes` must match CDE `type` values exactly. No explicit transformation
produces a binary or ordinal CDE yet, so callers need such variables in the
data model.

Date: 2026-11-14

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

### Decision: Code Binary and Ordinal Covariates as Integers

Status: Accepted

Context: Binary and ordinal CDEs are stored as text, and the regression
algorithms one-hot encoded every categorical covariate. A binary covariate
became a dummy column named after the alphabetically later level, and an
ordinal covariate lost its level order and became `K-1` dummy columns.

Decision: `linear_regression`, `linear_regression_cv`, `logistic_regression`,
`logistic_regression_cv`, `cox_regression_classical`, `cox_regression_stacked`,
`lmm`, `glmm_binary` and `glmm_ordinal` build their covariate columns with
`build_covariate_transformer` (`algorithms/exareme3/utils/covariate_encoding.py`),
which picks the coding from the CDE type in the worker metadata. `binary` is one
0/1 column: the first level declared in the data model is 0, the second is 1, and
the column is named `var[level coded 1]`. `ordinal` is one integer column
`0..K-1` following the declared level order, keeping the variable name.
`nominal` stays one-hot encoded with the first sorted level as reference,
including nominal variables with two levels. Binary and ordinal coding reuses the
private `_FederatedOrdinalEncoder` with the declared levels and
`handle_unknown="error"`, so a missing value or an undeclared level raises
`BadInputError` instead of being coded. Design matrix columns are ordered
nominal dummies, binary, ordinal, numerical. Cross-validation variants use the
same coding in every fold and in the final feature names.

Consequences: Coefficient names and counts change for models that use `gender`,
`alzheimerstatus`, `agegroup` or other binary and ordinal CDEs as covariates;
binary coefficients flip sign when the declared first level is the alphabetically
later one, and ordinal covariates now report one slope per level step. A
binary or ordinal covariate with fewer than two distinct observed values after
filtering (united across all workers) stops the analysis with a `BadUserInput`
naming the covariate and its single value, for example "Covariate 'gender' has
only one observed value (1) after filtering; it cannot be estimated. Remove it or
change the filters." There is no warning or drop path. The check runs when the
covariate transformer is fitted, so it covers all nine algorithms; the CV
algorithms fit it on the full filtered data before any fold runs, and a collapse
inside a single fold would also raise. Nominal covariates are not checked. Expected outputs of
`tests/algorithm_validation_tests/exareme3/expected/linear_regression_expected.json`
were regenerated for the 14 cases with such covariates.

Files affected: `exaflow/algorithms/exareme3/utils/covariate_encoding.py`, the
nine algorithms above, `tests/testcase_generators/design_matrix.py`,
`documentation/algorithms/`.

Date: 2026-11-14

### Decision: Store Categorical CDEs and Datasets as Integer Codes

Status: Accepted

Context: Nominal, ordinal and binary CDEs and the `dataset` CDE were VARCHAR
columns holding readable names, so DuckDB, the validator and the test data all
depended on free-text codes.

Decision: Only `continuous` is DOUBLE and `discrete` INTEGER; every categorical
CDE is an INTEGER column (`DType.from_cde` is `INT` for all non-continuous
types), with no text exceptions. `binary` codes are exactly `0` and `1`; `nominal`
and `ordinal` codes are integers from `1`; `dataset` codes are integers `1..N`
per data model. The enumeration label carries the readable name. `visitid` and
`dataset` stay integer-coded nominal variables. `subjectid` and `subjectcode` are
`discrete` whole numbers without enumerations, so subject identifiers are not
published as category levels (the longitudinal transformer's `visit1`/`visit2`
take visit codes such as `"1"`; the label, e.g. `BL`, is only a name). A categorical CDE without enumerations, or
with non-integer codes, is rejected by the exadata-validator and by the worker's
CSV loader (`_build_column_types`); there is no VARCHAR fallback. The
exadata-validator also rejects binary codes other than 0/1 and nominal, ordinal
or dataset codes below 1, both in `CDEsMetadata.json` and in the CSV values.
Inside the worker, `integer_codes_to_text` turns the integer columns back into
text codes right after the Arrow to pandas conversion, so algorithm parameters
(`positive_class`, `groupA`, `category_order`), metadata enumeration keys and
level-keyed results stay text codes (`"1"`, `"0"`) and nominal one-hot feature
names carry the code (`alzheimerbroadcategory[2]`).

API contract: `datasets` and `validation_datasets` are lists of dataset labels
(`"EDSD_0"`), the `label` of the `dataset` enumeration. The worker landscape
aggregator keys dataset locations and dataset variables by label, and
`GET /datasets`, `GET /datasets_locations` and `GET /datasets_variables` return
labels. A dataset reported by two workers under the same label is a duplicate
and is dropped. The controller translates labels to the integer codes of the
`dataset` enumeration when it builds the worker input, so workers and
algorithms still filter on codes, and results keep the `dataset` code. The
worker CSV loader rejects a data model that gives one label to two dataset
codes. Filter rules on categorical CDEs and on `dataset` must use integer
values. Algorithm parameters that name a level use the code as a string.

Migration: none. Existing deployed data is not migrated; only newly loaded
data uses the new types and integer codes. No migration script is provided.

Consequences: Every data model must be recoded and revalidated before loading,
and clients that send dataset or level names, or read result keys by name, must
send and read codes (labels come from the CDE metadata). Controller and workers
deploy together. Rollback restores the previous revision and the name-coded data.
The CSV file names are unchanged. Validation and prod fixtures were recoded
mechanically; outputs whose row order depends on level order need regeneration.

Files affected: `exaflow/datatypes.py`, the CSV loader, the UDF DB and service
layers, the controller type mapping, the algorithm specs, `exadata-validator`,
`tests/test_data`, and the validation and prod fixtures.

Date: 2026-11-16

### Decision: Compare Categorical Levels in Declared Order and Require positive_class

Status: Accepted

Context: `CommonDataElement.__eq__` compared enumerations as dicts, so workers
declaring the same levels in a different order were treated as compatible even
though the order defines ordinal ranks and binary 0/1 coding. The Cox regressions
also inferred the event level from 0/1 or false/true values when
`positive_class` was omitted, unlike logistic regression.

Decision: Categorical CDEs (nominal, ordinal and binary) compare enumerations in
declared order; a mismatch makes the data model incompatible and it is dropped
and logged by the worker landscape aggregator. The `dataset` CDE keeps its
exception: its enumerations may differ between workers. `positive_class` is
required for logistic regression (plain and cross-validated), the binary GLMM
and both Cox regressions and must be a level declared for the binary variable;
nothing is inferred. The specification mechanism cannot source the Cox enum from
`event_var` (it supports one `x` or `y` variable), so the level is validated in
`run()` before any UDF, with the same helper logistic regression uses
(`exareme3/utils/positive_class.py`).

Consequences: Requests that omit Cox `positive_class` are rejected. Heterogeneous
level order across workers removes the data model from the federation until the
order is aligned.

Date: 2026-11-16

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

## Separate logical requests from aggregation attempts

RabbitMQ redelivery may repeat a logical request. Preserve `request_id` for
tracing and use a fresh UUID `context_id` for each attempt, including all worker
aggregation operations and cleanup. Propagate both explicitly; aggregation logs
include both. This changes the aggregation RPC contract and requires coordinated
controller, worker, and aggregation-server upgrades. Regenerate protobuf bindings
using `exaflow/protos/README.md`.

## Isolate RabbitMQ replies and retain unsettled results briefly

Each delivery checks that its producer-created reply queue exists and publishes
through an isolated channel on the existing connection. Missing destinations and
broker nacks dead-letter only their request; transient failures retry the
serialized result. Completed replies can survive redelivery within one
controller for five minutes, bounded to 128 entries and 64 MiB. This remains
at-least-once execution without a durable result store. A sustained backlog can
reject HTTP analysis submissions under the intentionally shared capacity limit.

The broker is dedicated to Exaflow; no application queue-prefix allowlist or
request-body collision checks are added. RabbitMQ remains opt-in and externally
provisioned. The independent 30-second publication timeout is interruptible; the
20-second drain cap leaves ten seconds of lifespan shutdown headroom.
