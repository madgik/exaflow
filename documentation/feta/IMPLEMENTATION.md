# FETA Implementation Progress

Objective: implement the agreed FETA design, a simulated federation with mocked
aggregation, and package/API integration tests for `feta_pearson_correlation`.

## Milestones

- [x] Typed protocol, operation semantics, bounded context storage, expiry.
- [x] Separate worker and controller-driven aggregation services and clients.
- [x] FedTensor API, discovery, Pearson, and `/analysis` strategy integration.
- [x] Federation simulation, failure/security/lifecycle tests, real gRPC/API tests.
- [x] Regression validation and architecture/config/protocol documentation.

## Implementation Choices

- Separate FETA worker and aggregation gRPC services are registered alongside
  existing services. Existing Exareme3 protocols remain operational.
- Shared FETA types and lifecycle utilities live in `exaflow/feta`; algorithms,
  worker execution, and controller orchestration retain their repository layers.
- Existing Exareme3 data-loading helpers are reused in place.
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

The full statistics goal remains active. Next proposed algorithm is
`feta_ttest_independent`, pending the user's requested continuation checkpoint.
