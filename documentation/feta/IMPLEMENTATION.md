# FETA Implementation Progress

Objective: implement the agreed FETA design, a simulated federation with mocked
aggregation, and package/API integration tests for `feta_pearson_correlation`.

## Milestones

- [x] Typed protocol, operation semantics, bounded context storage, expiry.
- [x] Separate worker and controller-driven aggregation services and clients.
- [x] FedTensor API, discovery, Pearson, and `/analysis` strategy integration.
- [x] Federation simulation, failure/security/lifecycle tests, real gRPC/API tests.
- [x] Regression validation and architecture/config/protocol documentation.

Ownership note: the independence update at the end of this record supersedes
earlier milestone descriptions of helper reuse and shared inference. Historical
validation results remain records of those earlier implementations.

## Implementation Choices

- Separate FETA worker and aggregation gRPC services are registered alongside
  existing services. Existing Exareme3 protocols remain operational.
- Shared FETA types and lifecycle utilities live in `exaflow/feta`; algorithms,
  worker execution, and controller orchestration retain their repository layers.
- Data loading and computation helpers are now FETA-owned copies; legacy
  implementations remain unchanged.
- No runtime implementation was present at the start; the initial workspace
  was clean. No agent delegation is in use.

## Latest Validation

`uv run pytest -q tests/standalone_tests -m 'not smpc and not smpc_cluster' --disable-warnings --tb=short`:
**1,541 passed**, before the final FETA cancellation/dtype hardening cases.

`uv run pytest -q tests/standalone_tests/feta --disable-warnings --tb=short`:
**60 passed** on the final FETA implementation. This includes eight package
integration tests using the actual Quart app, DuckDB loader, and localhost gRPC
workers/aggregation services. Worker membership/metadata are supplied by a test
landscape; no Kubernetes or production deployment is involved.

Final focused regression command:

```bash
uv run pytest -q tests/standalone_tests/feta tests/standalone_tests/aggregation_server tests/standalone_tests/controller/services tests/standalone_tests/test_import_algorithm_modules_idempotent.py tests/standalone_tests/test_worker_run_udf_response.py tests/standalone_tests/exareme3/test_udf_service.py tests/standalone_tests/federated_algorithms/statistics/test_pearson_correlation.py --disable-warnings --tb=short
```

**226 passed** after all runtime changes. Numerical edge cases retain the
expected Pearson divide-by-zero/invalid-value warnings.

Protobuf regeneration was compared byte-for-byte with checked-in generated
files. Python compilation, `git diff --check`, all changed-file whitespace,
and local Markdown links passed. Existing Exareme3 helper files are unchanged.
Ruff/formatters were not run, following repository instructions.

## Completion Audit

| Requirement | Evidence |
| --- | --- |
| Separate FETA flow and discovery, existing analysis API | `FetaStrategy`, FETA registry/spec loading, real `/analysis` integration tests |
| Prepared aligned x/y, ordered columns, float64, omitted-x behavior | Worker initialization and rectangular/omitted-x parity tests |
| Single typed operation, fixed allowlist, literal arrays, scoped references | Protobuf source, worker validation, malformed-payload and dtype tests |
| Immediate tensor operations and automatic global reductions | Arithmetic/matrix/broadcasting simulations and fetch-before-worker-completion test |
| Independent controller prepare / workers submit / controller fetch | Real gRPC tests, blocked fetches, duplicate submission/fetch tests |
| Shared bounded dictionary, minimal locks, intermediate reuse | Concurrent store tests, resource tests, reference reuse assertions |
| Worker skipping and unchanged privacy threshold | Simulation and real filtered-worker skip tests |
| Failure without retries, cancellation, no late publication | Failed-worker, queued initialization, computation cancellation, and coroutine cancellation tests |
| Fixed maximum lifetime and background cleanup | Deadline/expiry, blocked-fetch wakeup, cleanup-job shutdown/health callback tests |
| Pearson output/formula/error parity | Original core comparisons, all matrix fields, constant and n=0/1/2/3 cases |
| Preserve Exareme3 and existing aggregation protocol | Full standalone suite, final focused regressions, both aggregation services on one real server |
| Protocol/config/API compatibility and rollback documented | Canonical FETA AGENTS, API/context docs, algorithm page |

The user-requested temporary missing-value rejection remains implemented, with
its explicitly requested TODO. It is not an unimplemented runtime placeholder.
Preprocessing, broader tensor layouts, and stronger authentication are future
scope documented in the canonical record.

Implementation and the required simulations/package integration tests are
complete. Production/Kubernetes deployment was not run because the validation
target was the mocked federation and package integration suite. Rollout requires
the updated controller, worker, and aggregation service/protobuf code together;
rollback details are in the canonical FETA record.

### Typed interface and deployment parity follow-up

The public abstract `FedTensor` interface now lives in `algorithms/feta/tensor.py`
and is exported from `algorithms/feta`. Runtime tensors inherit it, and algorithm
input annotations expose the supported tensor algebra to editors.

Added six deployment parity cases comparing both `/analysis` implementations
with identical complete-case selections. Production execution was attempted
but the controller at `127.0.0.1:5100` refused connections. This environment has
no Docker executable/socket, kubectl, Helm or Kind; Kubernetes deployment could
not be performed. Real gRPC/DuckDB/Quart integration checks remain available
through the standalone FETA suite, with tightened numerical parity tolerances.

### Development deployment parity validation — 2026-10-13

Ran `uv run inv create-configs` followed by
`uv run inv deploy --no-install-dep --no-structure-data`, using existing local
configuration and prepared worker data. Waited for `/datasets` to expose
`dementia:0.1` before validation; HTTP availability alone precedes landscape
initialization.

```bash
uv run pytest -q tests/prod_env_tests/test_feta_pearson_correlation.py tests/prod_env_tests/test_pearson_correlation.py --disable-warnings --tb=short
```

Result: **15 passed in 11.29 seconds**. Ten FETA parity cases cover all five
existing Pearson fixtures, with explicit and omitted `x`; five existing Pearson
tests also passed. Both implementations receive identical complete-case filters.
Normal Pearson retains its required missing-value preprocessing step, which
changes no rows after filtering; FETA receives no preprocessing. Counts, labels
and output structure match exactly; numerical results meet `rtol=1e-10`,
`atol=1e-12`. This verifies the real development services, not Kubernetes.

Corrected validation requests to include `value: null` in `is_not_null` filters,
as required by the shared API filter validator. No runtime changes were needed.

## Statistics migration: one-sample t-test

The active objective is to migrate all computation modules under
`algorithms/federated/statistics` to FETA, add NumPy-equivalent operations where
needed, consult the user for different operation contracts, and request
continuation after every completed algorithm. This milestone completes
`feta_ttest_onesample`; the broader migration remains active.

The new algorithm reuses the existing specification/response model in place,
registers independently with `AlgorithmType.FETA`, rejects preprocessing and
missing inputs, and computes count/sum/squared sum with existing tensor algebra.
The controller derives reference-centered moments and inference. The prepared
y group contains the selected sample; no x group is constructed. No new worker/protobuf operations
or deployment configuration were needed.

Validation:

```bash
uv run pytest -q tests/standalone_tests/feta tests/standalone_tests/federated_algorithms/statistics/test_ttest_onesample.py tests/standalone_tests/test_import_algorithm_modules_idempotent.py --disable-warnings --tb=short
uv run inv deploy --no-install-dep --no-structure-data
uv run pytest -q tests/prod_env_tests/test_feta_ttest_onesample.py --disable-warnings --tb=short
```

Results: **139 focused tests passed** and **5 live development-deployment parity
tests passed**. The latter used the existing prepared datasets, waited for worker
landscape readiness and algorithm discovery, and compared all response fields
against the existing runtime algorithm on identical complete-case selections.
Numerical tolerance is `rtol=1e-10`, `atol=1e-12`.

Preserved quirks: `mean_diff` is the sample mean, intervals are centered on that
mean, and constant-sample NaN/infinity behavior is retained. Registering the
additional algorithm is additive; existing algorithms and transport are unchanged.
Rollback consists of removing its FETA module and associated discovery entry
(which is automatic), tests and documentation; no stored-data migration is involved.

The paired t-test was the next migration milestone and is now complete, as
recorded below. Remaining statistics modules are still to be migrated.

## Specification-driven named inputs

Removed `resolve_inputs` from the base and both existing algorithms. The
controller selects declared groups from the specification and validated request,
initializes only those groups, and invokes `run(**tensors)`. One-sample t-test
declares `run(self, y: FedTensor)`. Pearson declares required y and optional x,
aliasing y inside the algorithm and using y labels when x is omitted.

`FetaRuntime.initialize(inputdata, columns)` now accepts a dictionary of selected
column groups and returns a dictionary of named tensors. This replaces its old
three-argument/tuple API; callers have been updated without a compatibility shim.
Worker responses omit unselected descriptor fields, and controller validation
checks exact group presence, float64 matrix shapes, selected column widths and
aligned row counts. Explicit identical groups still share one stored array.

The protobuf field layout and HTTP request/response shapes are unchanged. Older
workers reject single-group initialization, so deploy the updated controller and
workers together. Rollback must restore both initialization implementations and
algorithm signatures together. Privacy thresholds, cleanup and aggregation
semantics remain preserved.

Focused validation: **182 tests passed**, covering FETA, controller services and
algorithm discovery, including named y-only API calls, omitted-x Pearson,
unselected missing columns, mismatched descriptor presence/shape, shared storage
and cleanup races.

Live validation after `uv run inv deploy --no-install-dep --no-structure-data`:

```bash
uv run pytest -q tests/prod_env_tests/test_feta_pearson_correlation.py tests/prod_env_tests/test_feta_ttest_onesample.py --disable-warnings --tb=short
```

Result: **15 passed in 3.46 seconds**, covering explicit/omitted Pearson x and
the one-sample algorithm against their normal runtime counterparts.

## Statistics migration: paired t-test

Implemented `feta_ttest_paired` with typed named x/y inputs, the existing
specification and response model reused in place, independent FETA routing, and
no preprocessing. Paired differences are x minus y, preserving runtime direction.
A stored difference tensor feeds five immediate reductions: count, each sample
sum, difference sum and squared difference sum. Inference uses ordinary
controller NumPy/SciPy. Unreported sample standard deviations from the old core
are not computed. No new worker operators, protobuf changes or config are needed.

```bash
uv run pytest -q tests/standalone_tests/feta tests/standalone_tests/federated_algorithms/statistics/test_ttest_paired.py tests/standalone_tests/test_import_algorithm_modules_idempotent.py --disable-warnings --tb=short
uv run inv deploy --no-install-dep --no-structure-data
uv run pytest -q tests/prod_env_tests/test_feta_ttest_paired.py --disable-warnings --tb=short
```

Results: **208 focused tests passed** and **5 live development-deployment parity
tests passed**. Tests cover all alternatives, multiple alpha values, uneven/empty
partitions, privacy skips, small/constant samples, context/shape rejection, API
discovery and required-group validation. Every response field is compared to the
existing implementation with `rtol=1e-10`, `atol=1e-12`; t statistics and p-values
are additionally checked against SciPy.

This registration is additive. Existing algorithms and worker protocol are
unchanged. Rollback removes the new module, its tests and documentation; no data
or configuration migration is involved.

The independent t-test migration follows below; the full statistics goal remains
active, with a continuation checkpoint after each algorithm.

## Independent t-test and String Tensor Support

Implemented `feta_ttest_independent`, retaining the existing specification and
response model in place. Workers preserve text labels as fixed-width Unicode
arrays and numerical inputs as float64. Equality, single-column indexing and
boolean row filtering are allowlisted operations; `federated_numpy.equal` is
also available. Filtered tensors remain local and acquire shared selection
lineage when the same mask object is reused. No raw-array retrieval is added.

Six immediate sums supply group counts, sums and squared sums. NumPy/SciPy
controller inference preserves the existing pooled variance formulas, errors,
nonfinite edge behavior and one-sided infinity markers. Workers with empty
filtered groups remain participants.

Protocol source and generated bindings include string/boolean scalar operands,
array literal dtype, operation enum extensions and filtering lineage. Array dtype
is explicitly checked, including restoration of Arrow-encoded boolean literals.
Controller and workers must be deployed together; existing Exareme3 protocol,
configuration and datasets are unchanged. Rollback restores the prior FETA code
and bindings and removes the new registration; no data migration is needed.

Validation uses focused federation simulations, real Quart/DuckDB/gRPC package
integration and live development-tool deployment (no Docker/Kubernetes). Five
live cases compare every response field with normal `ttest_independent`, using
the same complete-case filters and `rtol=1e-10`, `atol=1e-12`. String validation
covers Unicode, empty partitions, different local string widths, mixed-group
rejection, memory limits, alignment and failure cleanup. Missing-value rejection
and initial privacy thresholds remain unchanged; no per-group privacy threshold
is introduced.

```bash
uv run pytest -q tests/standalone_tests/feta tests/standalone_tests/federated_algorithms/statistics/test_ttest_independent.py tests/standalone_tests/test_import_algorithm_modules_idempotent.py --disable-warnings --tb=short
uv run inv deploy --no-install-dep --no-structure-data
uv run pytest -q tests/prod_env_tests/test_feta_ttest_independent.py --disable-warnings --tb=short
```

In the restricted workspace, focused tests use `.venv/bin/python -m pytest` and
local gRPC tests require socket access. Development deployment sets
`UV_CACHE_DIR=/tmp/exaflow-uv-cache`, then waits for dataset discovery before the
parity tests. Ruff and formatters were not run.

Results: **256 focused regression/package tests passed**, followed by **58
focused checks passed** after final literal-size/index guards. **5 live
development-deployment parity tests passed** against the final protocol.
Changed Python syntax, local Markdown links, protobuf regeneration equivalence
and `git diff --check` passed.

The approved one-way ANOVA migration follows below. The full statistics migration remains unfinished.

## One-way ANOVA and Global Extrema / Unique

Implemented `feta_anova_oneway`, using category-label union and five immediate
reductions per category. Full ANOVA, Tukey, extrema and mean/std-bound outputs
retain the original response and formulas. The mathematical core's new
`fit_from_statistics` entry point reuses inference without Exareme3 execution;
legacy fit delegates to it and its regression suite passes.

Added typed `federated_numpy.min`, `max` and `unique`. Global unique initially
releases sorted string category labels, not distinct numerical observations.
Local NumPy unique supports numerical arrays. Min/max preserve distributed
placement for local axes and use their respective aggregation kinds for global
axes. Empty local subsets use neutral extrema; globally empty extrema fail.

Protocol additions include aggregation kind and bounded typed string-array
transport. Dynamic union shapes/dtypes are checked at worker acknowledgement,
server submission and controller fetch. Invalid submissions wake waiting fetches.
Controller, worker and aggregation service must deploy together; existing
Exareme3 protocol, dataset storage and configuration are unchanged. Rollback
restores FETA bindings/code, removes the registration, and restores the prior
core fit entry point. No data migration is involved.

```bash
uv run pytest -q tests/standalone_tests/feta tests/standalone_tests/federated_algorithms/statistics/test_anova_oneway.py tests/standalone_tests/test_import_algorithm_modules_idempotent.py --disable-warnings --tb=short
uv run inv deploy --no-install-dep --no-structure-data
uv run pytest -q tests/prod_env_tests/test_feta_anova_oneway.py tests/prod_env_tests/test_feta_ttest_independent.py --disable-warnings --tb=short
```

Results: **290 focused regressions and package tests passed**, followed by
**36 focused extrema/union/ANOVA checks passed** after final bound/skip cases.
**10 live development-deployment parity cases passed**: all five ANOVA fixtures
and all five independent t-test fixtures. Full nested responses use
`rtol=1e-10`, `atol=1e-12`; SciPy independently validates ANOVA F-tests.
No Docker/Kubernetes, Ruff or formatters were used.

The next proposed migration is `feta_anova_twoway`, pending the user's
continuation checkpoint. The full statistics migration remains unfinished.

## FETA Implementation Independence

Following the user's ownership decision, FETA copies the required implementations
instead of importing Exareme3 or shared algorithm/helper libraries. All five
algorithms now own specification definitions and Pydantic response models;
Pearson owns formatting, and ANOVA owns its copied inference/formatting module.
Worker/controller data selection, SQL loading/filters and Arrow serialization
also live in FETA. The shared ANOVA core is restored exactly to its prior code.

An import-boundary regression test permits FETA-owned production modules and
explicit public platform contracts only. Tests can still import normal algorithms
as numerical oracles. The common HTTP API, specification schema types, config,
registry, worker landscape, service interface, error types and protobuf/service
hosting remain platform integration points. No algorithm/math/data helper
implementation is shared between the paradigms.

Validation: **293 regression/package checks passed**, then **98 focused boundary,
specification and t-test checks passed** after direct FETA specification typing.
**30 live development-deployment parity cases passed** across Pearson, one-sample,
paired and independent t-tests, and one-way ANOVA. Production-like tests use the
development tools, with no Docker/Kubernetes.

HTTP results, numeric formulas, transport, config, privacy thresholds and cleanup
are unchanged. Internal response-model class identities are now FETA-owned.
The copied implementations must be maintained independently. No data migration
is needed. Rollback can restore only the FETA modules/imports; no legacy source
change is required or retained. The statistics migration checkpoint remains
`feta_anova_twoway`, which has not been started.

## Two-way ANOVA and Design Operations

Implemented `feta_anova_twoway` with owned specification, response model, copied
Type I/II table inference and controller-authored tensor expressions. No legacy
or shared algorithm source is modified or imported by production FETA code.
Observed factor levels use category unions, treatment masks become numerical
columns, and products form interactions. Five OLS designs preserve the normal
implementation's duplicate intercept, Gram pseudoinverse and rank conventions.
The controller computes coefficients/ranks from completed global matrix products;
workers calculate predictions and residuals. Eighteen aggregation steps supply
two unions, count and three summaries per model. Unused OLS metrics are omitted.

Added NumPy-equivalent `ones_like` and `column_stack` to `federated_numpy` and the
fixed worker allowlist. Distributed column stacking requires aligned vectors or
matrices partitioned on axis zero, rejects mixed numerical/text inputs and local/
distributed mixtures, and checks result allocation before execution. Ordinary
NumPy arrays execute locally. ONES_LIKE/COLUMN_STACK extend the protobuf operation
enum; controller and workers must update together. Aggregation schemas, config,
privacy thresholds and data storage are unchanged. Rollback removes the new
registration and restores FETA operation code/bindings; no data migration occurs.

```bash
uv run pytest -q tests/standalone_tests/feta tests/standalone_tests/federated_algorithms/statistics/test_anova_twoway.py tests/standalone_tests/test_import_algorithm_modules_idempotent.py --disable-warnings --tb=short
uv run inv deploy --no-install-dep --no-structure-data
uv run pytest -q tests/prod_env_tests/test_feta_anova_twoway.py tests/prod_env_tests/test_feta_anova_oneway.py --disable-warnings --tb=short
```

Results: **317 regression and package tests passed**, followed by **23 focused
checks passed** after factor-order and mixed-dtype guard coverage. **10 live
development-deployment cases passed**, comprising all five two-way ANOVA cases
and five one-way ANOVA regression cases. Full responses use `rtol=1e-10`,
`atol=1e-12`; simulations independently compare full-rank results with statsmodels
for both types. Other tests cover privacy skips, Unicode, empty partitions,
missing cells, saturated models, insufficient levels, alignment, memory bounds,
factor ordering and cleanup. No Ruff, formatters or Docker/Kubernetes were used.

The next proposed migration is `feta_chi_squared`, pending the user's
continuation checkpoint. The full statistics migration remains unfinished.

## FedColumns and Logistic Regression — 2026-10-14

Implemented the approved column-collection approach and standard
`feta_logistic_regression` before returning to statistics. Typed FedColumns owns
ordered references, immutable column metadata and row-count metadata. Worker
initialization loads a single aligned table and stores homogeneous columns;
native integer collection dtypes are preserved alongside unchanged float64
matrix inputs. Declarative column_inputs selects the representation without
input-resolution hooks. Metadata comes from validated engine CDE contracts.

Logistic regression preserves code-based feature names, categorical-first order,
one intercept, class-count checks, Newton updates, inversion/pseudoinverse policy,
50-iteration/1e-4 stopping rules and complete inference outputs. Text categories
use union; numeric categories use declared-code presence counts, rejecting
undeclared values without releasing arbitrary numerical uniques. Added fixed
NumPy-equivalent exp, log and where operations and typed public functions.

The initialization protocol adds column groups and named tensor descriptors.
Controller and workers must update together; HTTP requests/results, aggregation
schemas, config, privacy thresholds and data are unchanged. Internal input/model
implementations remain FETA-owned, with no legacy/shared-helper imports.
Rollback restores FETA modules/bindings and removes the new registration; no
legacy source change or data migration is needed.

Validation: **345 focused regression/package tests passed**. Simulations cover
numerical, text categorical and mixed/numeric-category covariates, uneven/empty
partitions, privacy skips, feature ordering, unknown codes, class errors,
nonconvergence, forced pseudoinverse fallback, typed reference reuse, metadata
immutability, dtype overlap, function semantics and cleanup. Statsmodels provides
independent coefficient checks on identifiable designs. A real package API test
uses mixed covariates through DuckDB and gRPC.

**10 live development-deployment cases passed**: all five standard logistic
fixtures and all five two-way ANOVA regression fixtures, comparing full responses
with rtol=1e-10/atol=1e-12. A collinear-design check identified the normal solver's
unstable-inverse behavior; fallback correctness is tested with inversion failure,
and the numerical sensitivity is documented rather than changing solver policy.
No Docker/Kubernetes, Ruff or formatters were used.

The completed priority is standard logistic regression, not its cross-validation
variant. The statistics goal remains unfinished; next proposed algorithm is
feta_chi_squared, pending the user's continuation checkpoint.

Final ownership/input checks: **48 package/logistic checks passed**, followed by
**28 targeted logistic/function/boundary checks passed** after nullable integer
conversion. Changed Python syntax, Markdown links, exact protobuf regeneration
and git diff --check passed. Legacy Exareme3 and shared federated source have no
diff. Numeric metadata code conversion is explicit for int/real/text SQL types;
optional column groups follow selected-request presence.
