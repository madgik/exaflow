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
are VARCHAR (`CATEGORICAL_CDE_TYPES`). Metadata dicts, the worker proto,
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
