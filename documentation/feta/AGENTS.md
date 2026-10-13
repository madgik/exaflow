# FETA Architecture and Agent Guidance

Initial decision record: **2026-10-13**.

This is the canonical, evolving architectural record for FETA across algorithms,
controller, and workers. Read it alongside the repository's
[root agent instructions](../../AGENTS.md) before working on FETA. Keep FETA's
architectural documentation here; other repository guidance should link here
rather than duplicate these decisions.

The initial runtime is implemented with the contracts below. Keep
confirmed decisions, implementation proposals, and unresolved questions
separate. Update the record as discussion continues; do not silently promote a
proposal to a confirmed decision.

## Confirmed Decisions

### Identity, Ownership, and Execution

- The paradigm is named **FETA**. Use `feta` for its namespaces rather than
  naming it Exareme4. The module layout is recorded under Implementation Contracts.
- FETA has a separate execution flow and orchestration, while continuing to
  use the existing aggregation server process through its separate FETA service.
- FETA owns its algorithm specifications, response models, mathematical helpers,
  data selection, data loading, filter construction and serialization. Copy
  required implementations into FETA; do not import Exareme3 or shared algorithm
  libraries, and do not modify those implementations to support FETA. This
  supersedes the earlier decision to call helpers in place.
- Existing public platform contracts still integrate FETA with the engine:
  specification schema types, configuration, registry, worker landscape, service
  interface, protobufs and API error types. They are not computation helpers.
  Third-party NumPy/SciPy/pandas/Pydantic libraries remain supported.
- Tests may import the legacy implementations as independent parity references;
  FETA production code cannot depend on those reference imports.
- Algorithm developers write tensor expressions in the controller-side
  algorithm. FETA system engineers control the supported local operations and
  their implementations; algorithm developers compose those operations.
- Execute local operations immediately, sending exactly one operation per
  worker request. Do not batch multiple local operations into one request or
  automatically batch successive immediate reductions.
- Exareme3-specific algorithm scaffolding does not define FETA's implementation
  workflow.

### API Registration and Execution Routing

- Use the existing `/analysis` endpoint and shared specification models. Add
  `AlgorithmType.FETA` to identify FETA algorithms.
- Discover/register FETA algorithms independently and route their registered
  type to a dedicated FETA execution strategy before context initialization.
  An algorithm registered in FETA executes through FETA, without an Exareme3
  fallback or an additional caller-selected execution mode.
- Initially, add FETA algorithms under distinct names so they can coexist with
  existing algorithms in the shared name-keyed specification registry.
- The first algorithm ID is `feta_pearson_correlation`. Keep the existing
  Exareme3 `pearson_correlation` available under its current name, allowing
  comparisons through the same API.
- The caller selects the implementation through `algorithm.name`; routing
  follows the registered algorithm type rather than inferring it from a name
  prefix.
- Use a FETA-specific algorithm base class and registry keyed by specification
  name. Discovery imports FETA algorithm modules, and shared specification
  loading registers their specifications with `AlgorithmType.FETA`. Duplicate
  names fail explicitly. `FETA_ALGORITHM_FOLDERS` configures discovery, defaulting
  to `./exaflow/algorithms/feta`; imports reuse the existing idempotent importer.

### Context Initialization and Algorithm Inputs

- The runtime prepares only the selected input groups declared in the
  specification and passes them to `run` by name. They are already
  `FedTensor` objects, or `FedColumns` for declared tabular groups; the algorithm
  does not cast them or select initial input columns
  through a `self.tensor(...)` call.

- Each algorithm declares its typed named inputs, for example `run(self, y: FedTensor)` or `run(self, y: FedTensor, x: FedTensor | None = None)`. There is
  no `resolve_inputs` hook. Omitted optional groups are not loaded or passed;
  ordinary Python defaults inside `run` define algorithm-specific behavior.

- The prepared tensors must preserve the requested variable order and refer to
  aligned local rows on the same participating workers, so expressions such as
  `y.T @ x` operate on corresponding observations.

- During initialization, skip workers whose selected local data falls below
  their existing privacy minimum-row threshold, matching current behavior.
  Fail the analysis if no workers qualify.

- Establish the eligible worker set before configuring the aggregation server.
  Configure aggregation for that set and construct the initial `FedTensor`
  references from those workers only.

- Release any context state allocated on skipped workers. Skipping for
  insufficient rows does not authorize swallowing other initialization errors;
  missing values still follow the agreed rejection policy.

- Initially support only the `x` and `y` input groups. Defer additional named
  groups beyond the first version.

- Load initial numerical matrix tensors as `float64`, matching the current Pearson
  computation. This does not change the integer return type of `len(x)` or
  settle the complete set of supported literal operand dtypes.

- For `feta_pearson_correlation`, omitted `x` is handled inside `run` by aliasing
  the supplied `y` tensor. No second group is loaded and output labels use `y`
  for both dimensions. Explicit identical selections share a stored tensor.
  This is a Pearson input rule, not a universal fallback for future algorithms.

Initialization proceeds as follows:

1. The controller selects input groups from the specification and validated
   request, preserving column order, and determines candidate workers.
1. It sends an `InitializeContext` lifecycle request with the context ID,
   validated data selection, and input column groups.
1. Each worker loads the required columns once, in one DuckDB query applying
   dataset selection and filters, using the FETA-owned
   [data loader](../../exaflow/worker/feta/data_loader.py) and SQL filter builder.
   Their initial implementations are copied from the existing loader/helpers;
   the source implementations remain unchanged.
1. The worker checks the privacy threshold and the temporary missing-value
   policy, then constructs the selected groups from the same loaded rows, restoring
   each group's requested column order.
1. The worker stores the prepared arrays and returns references and metadata;
   workers with insufficient data are skipped under the agreed policy.
1. The controller creates a dictionary of named `FedTensor` inputs for eligible
   workers and invokes `run(**tensors)`. A y-only algorithm receives only `y`.

The initialization flow and protobuf schemas are implemented in `feta.proto`.

### Worker Storage and Lifecycle

- Preserve worker-local inputs and intermediate results across operation
  requests in an in-memory key-value store:
  `store[context_id][tensor_id]`.
- Use a normal Python dictionary shared by request threads within a worker
  process. Each worker process owns its own store.
- Keep locking minimal: use a short-lived store lock where concurrent context
  creation, deletion, or iteration must be coordinated. In particular,
  additions and deletions of different context IDs must not race with compound
  store operations or cleanup scans. Do not hold this lock during NumPy
  computation or network calls; execution on different contexts remains
  concurrent. Each operation holds references to immutable inputs and to its
  original context object. Result publication checks that the same unexpired
  object is still registered under the context ID. Cleanup removes the entry
  and clears its stored values without waiting for NumPy computation.
- Workers generate IDs for new results. Requests do not supply an output ID.
- Reuse stored intermediate results. Detailed caching policy remains open.
- At execution end, clean up each worker's context and all its associated
  values, including failure paths. Also clean up the aggregation-server
  context. Cleanup cancels aggregation waits before releasing worker contexts.
- Use the FETA `context_id` as the aggregation-session identifier as well as
  the worker-store context identifier. The external analysis request ID can
  remain separate for logging.
- The controller assigns aggregation step numbers, starting at `1` and
  increasing for each aggregation within the context. It supplies the same
  step number to all participating workers and to the aggregation server.
  Local computations do not consume aggregation steps.

### Background Cleanup

- Use controller-triggered cleanup on success or failure as the normal path.
  Automatic expiry is the fallback for contexts abandoned when the controller
  cannot clean up.
- Run one background cleanup thread per worker process and one in the
  aggregation-server process, alongside their synchronous gRPC servers. Start
  it once during service startup, not per request, and keep it independent of
  the gRPC request-thread pool.
- Use a stop event with an interruptible timed wait between scans. At service
  shutdown, signal the event and join the cleanup thread. A daemon thread may
  be used, but it does not replace explicit shutdown handling.
- Both controller requests and background expiry use the same cleanup
  implementation. Repeated cleanup is harmless. Coordinate cleanup with
  active operations so late results cannot recreate a closed context.
- The cleanup thread accesses the same process-local context store under the
  agreed minimal synchronization. Make cleanup-thread failures visible through
  logging and health reporting.
- Use a configurable maximum context lifetime in minutes, measured from
  context creation. Activity and new aggregation steps do not reset it. No
  heartbeat or lease renewal is required in the initial version.
- Initial implementation default: **60 minutes**. This value is a chosen
  default, not a user-specified duration; the confirmed policy is a maximum
  lifetime in minutes. Configure it through the existing configuration layer.
- Track elapsed lifetime using each process's monotonic clock. When the
  deadline is reached, the context is expired even if the execution is still
  active; close it and remove its stored values through the cleanup path.
  The background scan interval defaults to 30 seconds. Access and result
  publication also check the deadline, so expired data cannot be reused between
  scans. Normal cleanup occurs immediately at execution end.
- Expiry must terminate a waiting aggregation fetch with an error and prevent
  late operations from storing results into the expired context. It does not
  forcibly stop running NumPy computations; arrays still referenced by those
  computations can only be released after their active users finish.

### Failure and Retry Policy

- If a participating worker fails after initialization, fail the entire
  execution. Do not silently remove that worker and continue with a different
  population. Skipping insufficient-data workers is an initialization behavior.
- Do not retry failed operations automatically in the initial version. A lost
  response may follow successful local execution or contribution submission;
  it is not evidence that replay is safe.
- On execution failure, terminate any pending aggregation wait and attempt
  cleanup of all initialized worker contexts and the aggregation-server
  context. Removing the aggregation session marks it closed and notifies its
  condition variable, waking a blocked fetch with a context-closed error.
- Cancelling the controller coroutine triggers context cleanup even while its
  synchronous execution runs off the event loop. RPC futures are cancelled
  before cleanup, and new execution requests on closing clients are rejected.
  Workers check RPC cancellation before loading or publishing data, so queued
  initialization cannot repopulate a context after cancellation. Workers still
  executing native code cannot publish
  into closed contexts when that code finishes.
- Automatic expiry for abandoned contexts is confirmed through the background
  cleanup approach above and is implemented on workers and aggregation services.

### Controller-Driven Aggregation Flow

The controller knows when an expression requires aggregation and explicitly
coordinates each aggregation:

1. The controller prepares the aggregation server to receive contributions
   from the participating workers for that aggregation.
1. After setup succeeds, the controller triggers the workers to send their
   local contributions directly to the aggregation server.
1. The controller fetches the global result directly from the aggregation
   server. The fetch waits until the result is available, or an error occurs.

Workers are contribution producers; the controller is the consumer of the
global result. Workers do not need to receive and relay the global result to
the controller. Submission and result retrieval are separate responsibilities.
The server must retain a completed result until the controller can retrieve it,
including when contributions arrive before the fetch begins, unless the context
is explicitly cleaned up or reaches its maximum lifetime.

The controller returns the completed result to algorithm code as the agreed
NumPy array or concrete scalar. This remains automatic from the algorithm
developer's perspective; no explicit federated-sum API is introduced.

Determining eligible workers during initialization is separate from preparing
each individual aggregation. The implemented setup, submission, and fetch
messages all identify the aggregation by the shared context ID
and controller-assigned step number.

### Tensor Model and Automatic Aggregation

- `FedTensor` is the controller-side representation of a distributed value. It
  contains worker-specific tensor references and relevant shape, dtype, and
  partitioning metadata.
- Workers hold actual arrays. `LocalTensorRef` identifies a stored tensor
  within a worker execution context. Local tensor IDs may differ between
  workers for the same logical `FedTensor`.
- Use NumPy-style numeric axes, initially `0` and `1`. Do not introduce
  named-axis metadata. Internal partitioning metadata is still necessary.
- Do not introduce a `SharedTensor` abstraction. Completed global calculations
  return ordinary NumPy arrays or concrete scalars; subsequent calculations on
  those values use ordinary NumPy/SciPy.
- Do not introduce a `Scalar` wrapper. Supported ordinary Python values may be
  operation inputs.
- Ordinary numeric NumPy arrays may also be inputs alongside `FedTensor`
  references. Send the same concrete array as an operation input to each
  participating worker and apply NumPy broadcasting locally. Do not introduce
  a shared-state tensor wrapper for these values.
- Infer aggregation from operation semantics and partitioning. There is no
  separate developer-facing federated-sum API.
- Local results that retain partitioned dimensions remain `FedTensor` values.
  Additive partial contributions to the same output elements are summed through
  the aggregation server.
- Transpose updates partitioning metadata. Contracting a partitioned dimension
  during matrix multiplication triggers aggregation.
- Validate that contracted distributed dimensions represent aligned
  partitions; compatible shapes alone are insufficient.
- Implement `len(x)` for the initial row-partitioned input case by summing
  worker-local lengths and returning a Python integer.
- Use standard `.T` and `@`. Do not implement a custom `cross_product()` API.

For initially row-partitioned `x` and `y`:

| Expression | Execution | Result |
| --- | --- | --- |
| `x * 2` | Local elementwise multiplication | `FedTensor` |
| `x.T` | Local transpose; partitioning moves to axis `1` | `FedTensor` |
| `x.sum(axis=1)` | Local reduction over columns | `FedTensor` |
| `x.sum(axis=0)` | Local sums followed by aggregation | NumPy array |
| `y.T @ x` | Local matrix products followed by aggregation | NumPy array |
| `len(x)` | Local lengths followed by aggregation | Python integer |

`y.T @ x` produces shape `(y_features, x_features)`, matching the existing
Pearson expression `np.einsum("ji,jk->ki", x, y)`. Each worker computes its
matrix product locally; the runtime recognizes the additive partial matrices,
aggregates them, and returns the complete result. No custom global cross-product
implementation is required.

For example, a global mean can be reused directly in a local expression:

```python
means = x.sum(axis=0) / len(x)  # Ordinary NumPy array on the controller.
centered = x - means  # Send means as an input to each worker; return FedTensor.
```

### Local Operation Protocol

Use typed protobuf messages over gRPC to transport operations, tensor
references, parameters, and literal scalar/array inputs. The transport choice
and message schemas are implemented in `feta.proto`.
Use the FETA-owned Arrow tensor byte encoding in
[serialization.py](../../exaflow/feta/serialization.py), initially copied from
the existing implementation, for numeric arrays carried in protobuf byte fields. Worker-side semantic
validation is still required.

Use one `LocalOperation` class for every operation, containing:

- An operation type.
- Inputs consisting of local tensor references, supported ordinary Python
  values, or numeric NumPy arrays.
- Operation-specific parameters, such as a numeric axis.

The operation class uses typed fields for its parameters (currently `axis`):

```python
LocalOperation(
    type=OperationType.MULTIPLY,
    inputs=[LocalTensorRef("tensor-a81f"), 2.0],
)
```

Do not introduce separate operation subclasses. Transpose, matrix
multiplication, and local reductions also use this same class with their
corresponding operation types.

For operations whose results remain local, execution proceeds as follows:

1. The controller resolves each `FedTensor` input to that worker's reference.
1. It sends one operation with the execution context.
1. The worker validates and dispatches to an approved implementation.
1. The worker stores the result and returns its reference and permitted
   metadata.
1. The controller constructs a new `FedTensor` from the worker references.

When an expression requires aggregation, use the controller-driven aggregation
flow above: prepare the aggregation server before triggering contribution
submission, then fetch the completed result directly at the controller.
Local computations still use approved `LocalOperation` instances.
`ExecuteRequest.aggregation_step` is zero for a local result or the prepared
step number for an additive partial. Workers independently verify that this
instruction matches the operation's actual placement before submitting.

Requests describe operations; they do not carry executable Python, import paths,
pickled functions, or arbitrary NumPy function names. Serialize operation data,
not executable Python objects. Operation dispatch is controlled by FETA
engineers; algorithm developers cannot install handlers through requests.

An operation allowlist constrains executable behavior but does not alone
establish privacy for every composition. Preserve existing privacy boundaries,
including minimum-row checks. Do not provide unrestricted retrieval of local
arrays. Context IDs provide namespacing, not caller authorization.

### Initial Algorithm Scope and Compatibility

- No preprocessing in FETA initially.
- Temporarily reject missing input values with a clear error. **TODO:** revisit
  this temporary policy; include an explicit TODO alongside the eventual
  implementation. Do not silently drop incomplete rows or allow missing values
  to propagate instead of this agreed rejection.
- Pearson, registered as `feta_pearson_correlation`, is the first target.
  Preserve output structure, matrix orientation,
  errors, coefficient clipping, constant-feature behavior, p-values, and
  confidence-interval formulas, allowing small floating-point differences.
- Preserve the existing Pearson calculation quirks described below during
  migration. Correcting them is separate work.

### Statistics Migration

- Migrate every computation module under `algorithms/federated/statistics` into
  FETA, using distinct `feta_` algorithm names where there is a runtime algorithm.
- Add required NumPy-equivalent operations to `federated_numpy`. Ask the user
  how to proceed when a required computation has no NumPy equivalent or requires
  a new contract beyond the agreed tensor language.
- After implementing and validating each algorithm, ask the user to continue
  before beginning the next. Keep the full migration objective active.
- Completed runtime algorithms: `feta_pearson_correlation`,
  `feta_ttest_onesample`, `feta_ttest_paired`, `feta_ttest_independent`, `feta_anova_oneway` and `feta_anova_twoway`. Remaining statistics modules are tracked in the
  [computation catalogue](LOCAL_COMPUTATIONS.md); they are not implicitly migrated.
- One-sample t-test prepares only the required sample `y` and its `run(self, y)`
  receives one stored tensor. It uses three global sums
  (length, sample sum, squared sum), with inference on the controller.
- Preserve its existing `mean_diff` field as the sample mean and its confidence
  intervals as intervals for that mean. One-sided infinity markers, constant
  sample behavior and the insufficient-observation error remain unchanged.
- Its public parameters, numeric input contract and response fields match the
  existing one-sample algorithm through FETA-owned definitions, with no
  preprocessing. Missing inputs remain rejected under the current FETA policy.
- Paired t-test receives required named x/y inputs and computes x minus y,
  preserving the existing runtime's direction. The difference tensor is stored
  and reused. Five immediate global reductions supply count, sample sums,
  difference sum and squared difference sum; inference runs on the controller.
  Its response, one-sided infinity markers and constant-difference behavior are
  preserved. No new worker operation or automatic batching is introduced.

The missing-value rejection is an explicit initial FETA limitation: there is no
FETA preprocessing path to reproduce Exareme3's selected preprocessing behavior.

## Current-Code Observations

These observations explain the design discussion; they do not require FETA to
inherit Exareme3's execution structure.

- The current Pearson wrapper dispatches one worker UDF. Its statistical core
  computes sufficient statistics and performs six aggregation calls.
- Exareme3 recreates its aggregation client for each UDF invocation, and the
  client's step counter starts at `1`. That caused the earlier concern about
  repeated controller-driven calls. It is not an inherent FETA limitation:
  FETA must maintain the agreed aggregation sequence across its execution.
- Exareme3 has an AST-based lazy aggregation facility. FETA's initial immediate,
  one-operation-per-request design does not depend on it.
- The current aggregation server's `Aggregate` RPC accepts a worker's
  contribution, waits for the global result, and returns it to that worker.
  Its step-completion bookkeeping expects consumption by the workers. There
  is no controller result-fetch RPC. FETA's confirmed producer/consumer flow
  required the separate `FetaAggregation` service now registered alongside the
  original service; Exareme3's existing protocol remains operational.
- The [worker gRPC server](../../exaflow/worker/grpc_server.py) uses a
  `ThreadPoolExecutor`; a worker process is not single-threaded. Worker config
  normalizes `grpc.max_workers` to a fallback of `10`, and the
  [Helm local-worker template](../../kubernetes/templates/exaflow-localworker.yaml)
  sets `GRPC_MAX_WORKERS` to `20`. These are source/configuration observations,
  not a measurement of a running deployment. The aggregation server also uses
  a thread pool. Any FETA context store must account for concurrent requests
  and cleanup.

### Pearson Quirks to Preserve

The [current wrapper](../../exaflow/algorithms/exareme3/statistics/pearson_correlation.py)
labels `alpha` as a confidence level and defaults it to `0.95`, whereas the
[statistical core](../../exaflow/algorithms/federated/statistics/pearson_correlation.py)
uses it as a significance level in `norm.ppf(1 - alpha / 2)`. Preserve the
existing calculation during migration rather than silently correcting the
interpretation.

The core rejects zero observations and insufficient observations at `n <= 2`,
but accepts three observations despite the confidence-interval standard error
`1 / sqrt(n - 3)` being singular. It also sets zero-denominator correlations to
zero, clips coefficients to `[-1, 1]`, and handles perfect correlations in its
p-value calculation. Preserve these behaviors and their output handling during
migration; they need focused regression coverage when FETA is implemented.

## Implementation Contracts

### Modules and Protocol

- `exaflow/algorithms/feta`: algorithm base class and
  `feta_pearson_correlation`. Specifications, response models and result
  formatting are FETA-owned copies; tensor expressions and calculation run in FETA.
- `exaflow/algorithms/feta/federated_numpy`: public typed NumPy-style function
  package. The initial functions are `sum(a, axis=None)` and `transpose(a)`.
- `exaflow/controller/services/feta`: execution strategy and per-analysis
  runtime. Synchronous gRPC orchestration runs off the Quart event loop.
- `exaflow/worker/feta`: independent operation executor and input preparation.
- `exaflow/feta`: tensor API, shared operation contracts, clients, settings,
  context storage, and cleanup job.
- `exaflow/protos/feta/feta.proto`: separate `FetaWorker` and `FetaAggregation`
  services on the existing worker and aggregation-server processes/ports.

Worker RPCs are `InitializeContext`, `Execute`, and `Cleanup`. Initialization
returns worker identity and input descriptors; execution returns a descriptor,
never local array data. Each descriptor includes tensor ID, shape, dtype,
numeric partition axis, and row-layout identity. The row layout originates at
initialization and is propagated by approved operations.
Requests may select either or both of the existing x/y groups. Responses include
only the selected descriptor fields; omitted groups have no protobuf message
presence and no stored array. The controller validates exact group presence,
numerical/Unicode matrix shapes, selected column counts and aligned row counts. The wire
schema is unchanged, but controllers sending y-only requests require workers
with the updated initialization semantics; deploy those services together.

Aggregation RPCs are `Prepare`, `Submit`, `Fetch`, and `Cleanup`. Preparation
specifies the exact worker set, output shape/dtype, aggregation kind, context,
and step. Union preparation uses a candidate-count upper bound instead of an
exact result length. Submission
acknowledges receipt without waiting for other workers. Contributions from
unexpected workers, duplicates (including after result completion), incorrect
steps, or inconsistent shapes/dtypes fail explicitly. Sum/min/max combine
contributions in stable worker-ID order; union returns sorted distinct labels.
Results are retained until fetch and released after the single
successful fetch. Fetch errors on expiry, cleanup, client cancellation, invalid
contributions, or the configured aggregation wait timeout.
Only one fetch may wait on a given step; concurrent duplicate fetches are
rejected without consuming another waiting slot or replacing the first result.

The controller starts fetch after dispatching worker requests and monitors the
fetch and worker futures together. Any failed future cancels the aggregation;
successful completion requires both the global result and valid worker
acknowledgements. The aggregation process reserves `max_contexts` extra RPC
threads above `max_grpc_connections`, leaving submission/control capacity
available while result-fetch calls block.

Regenerate the Python protocol from its source; never hand-edit generated files:

```bash
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. exaflow/protos/feta/feta.proto
```

### Operations and Limits

The initial allowlist is `ADD`, `SUBTRACT`, `MULTIPLY`, `DIVIDE`, `POWER`,
`TRANSPOSE`, `SUM`, `MATMUL`, `LENGTH`, `EQUAL`, `COLUMN`, `FILTER`, `MIN`, `MAX`, `UNIQUE`, `ONES_LIKE`, `COLUMN_STACK`, `EXP`, `LOG` and `WHERE`. Literals are Python numeric scalars or
NumPy arrays with integer dtypes up to 64 bits, `float32`/`float64`, or booleans.
String and boolean scalar literals are also supported. Numerical inputs loaded
from DuckDB remain `float64`; text groups use fixed-width NumPy Unicode arrays.
Object, complex, non-native-endian and higher-rank arrays remain rejected.
String-array literals are unsupported because the current Arrow tensor operand
encoding is numerical. Input groups cannot mix text and numerical columns.
Unicode allocation uses actual maximum label width and is bounded before array
construction; empty text partitions retain string dtype. Initial descriptor
validation accepts float64 or Unicode rather than assuming all inputs numeric.

Equality (`==` or `fnp.equal`) returns distributed boolean tensors. Numerical
reductions on booleans count true values; strings cannot be summed or used in
arithmetic/matrix products. `x[:, column]` selects one column from a
row-partitioned matrix, returning a one-dimensional tensor. `y[mask]` accepts an
aligned one-dimensional boolean FedTensor and returns a row-partitioned result.
Other indexing forms are unsupported. Tensor truth-value conversion is rejected.

The controller assigns selection lineage per mask object and sends it in the
FILTER request. Applying the same mask to multiple aligned tensors gives their
results the same lineage. Independently created masks conservatively produce
different lineage. Filter result row counts are data-dependent: workers report
actual shapes and the controller checks bounds, remaining dimensions, dtype,
partition and lineage. Workers with empty subsets remain participants.

These protocol extensions add string/boolean operand fields, explicit array
literal dtype, three operation values and selection lineage. Arrow stores boolean
tensor literals as uint8; workers validate 0/1 values and restore their logical
boolean dtype using the declared dtype. Deploy FETA controller and workers together.
Existing Exareme3 RPCs, routes, configuration and numerical aggregation schemas
are unchanged. Raw labels/masks are not released to the controller. The operation
allowlist does not establish privacy for every possible composition; existing
trust and authorization limitations continue to apply.

Supported tensors have rank at most two. Elementwise broadcasting cannot
broadcast over a distributed record dimension or align different record
layouts. `sum(axis=None)` contracts every dimension; explicit axes must be
nonnegative and in range. Matrix multiplication supports aligned distributed
contraction (`y.T @ x`), a row-partitioned matrix times a literal matrix, and a
literal matrix times a column-partitioned matrix. Other placements fail
explicitly. `len(x.T)` returns the replicated first dimension locally; length
on a tensor partitioned along axis zero aggregates its local first dimensions.

Store values are immutable. Reusing an intermediate `FedTensor` reuses its
worker references without recomputation. There is no expression deduplication,
automatic reduction cache, or automatic operation retry.

All three service configurations accept a `[feta]` section. Defaults are:

| Setting | Default |
| --- | --- |
| `context_lifetime_minutes` | 60 |
| `cleanup_interval_seconds` | 30 |
| `max_contexts` | 32 |
| `max_context_bytes` | 268435456 (256 MiB) |
| `max_tensor_bytes` | 67108864 (64 MiB) |

Each context's initial deadline is process-local and monotonic. Configuration
must be positive and finite; count/byte limits must be integers. Workers check
predicted output size before evaluation, initial tensor size before constructing
input arrays, and context capacity before storing results. The aggregation
service also bounds retained contributions and output allocations. Temporary
DuckDB loading and NumPy computation may use additional transient memory.
The controller also checks its execution deadline before tensor operations and
before returning the algorithm result. Expired contexts fail execution even if
controller-side postprocessing has finished without another worker call.

### Trust and Compatibility

FETA uses the existing deployment's controller-to-worker trust boundary and
gRPC transport; it does not introduce a new authentication system. Context and
worker IDs provide scope/identity checks, not cryptographic authentication.
Do not expose these RPCs directly to untrusted external callers. Worker-side
operation validation prevents arbitrary code execution, but arbitrary
compositions of allowed operations are not a differential-privacy guarantee.
Minimum-row checks and existing privacy configuration are preserved.

Existing HTTP request and Pearson response shapes are unchanged. The catalog
adds type `feta` and algorithm ID `feta_pearson_correlation`; consumers with
closed backend-type enums must accept `feta`. FETA rejects preprocessing,
missing inputs, validation datasets, and optional execution flags initially.
Existing Exareme3 services and algorithms continue using their original RPCs.
Roll out the generated FETA protocol and services together on controller,
workers, and aggregation server. Rollback disables/removes the FETA algorithm
and restores the prior service images/configuration; existing algorithms do
not require request migration.

## Federated Local Computation Library Proposal

The [federated local computation catalogue](LOCAL_COMPUTATIONS.md) reviews the
full `algorithms/federated` tree, retaining the original 17-module statistics
review and extending it to training, prediction, mixed effects, survival,
preprocessing, composition and cross-validation. It maps sources to current
operations, proposed kernels and required global reductions, and identifies hard
problems around local optimization, risk sets, cluster ownership, row lineage,
private output placement and intermediate memory. These are proposals and source
observations, not approved operations, expanded preprocessing/missing policies,
privacy releases or implemented algorithm migrations.

## Validation and Future Scope

Simulation tests use real operation/protobuf/storage code with an independent
mock aggregation server and uneven worker partitions. Package integration tests
use the actual Quart application, DuckDB loader, and localhost gRPC services;
worker membership/metadata are supplied by a test landscape. Tests cover Pearson
parity, rectangular orientation, omitted `x`, constant and small-sample cases,
input rejection, worker skipping, broadcasting, cleanup, expiry, duplicate
contributions, resource limits, and failure without retries.

```bash
uv run pytest -q tests/standalone_tests/feta
```

Full deployment/Kubernetes validation is separate from these integration tests.
Future work may add preprocessing, additional input groups, broader layouts and
operations, stronger authentication/data-release policies, retry deduplication,
or caching. Those extensions are not part of the initial runtime requirements.

## Superseded Ideas

Do not reintroduce the following earlier suggestions as accepted architecture:

- Worker-UDF-authored algorithms instead of controller-authored expressions.
- Named axes, a `SharedTensor` abstraction, or a `Scalar` wrapper.
- Deferred local expression graphs as the initial execution model.
- Multi-operation request lists or controller-supplied output IDs.
- A separate class for each local operation.
- A custom `cross_product()` or explicit federated-sum API.
- References such as the string `"x"` standing in for actual local tensor IDs.
- Requiring algorithm code to create its initial `x` and `y` tensors with
  `self.tensor(columns=...)` instead of receiving prepared tensors in `run`.
- Using Exareme3's worker-result-return flow for FETA aggregation. FETA workers
  submit contributions and the controller fetches the global result directly.
- Controller heartbeats or renewable leases for initial context expiry. Use
  the configured maximum lifetime from context creation instead.
- Moving or modifying Exareme3/shared algorithm helpers for FETA, or calling
  them in place. Copy needed implementations into FETA-owned modules instead.

## Documentation Maintenance

Keep confirmed decisions consistent with the discussion and mark proposals as
such. Verify links when moving or extending this record. Runtime, public API,
protobuf, dependency, and deployment changes require their own implementation
work and validation; this documentation record does not itself make those
changes.

### Public Typed Tensor Interface

Algorithm authors import the function package with
`from exaflow.algorithms.feta import federated_numpy as fnp`; the alias is optional.
Keep arithmetic, equality comparisons, and matrix multiplication
in operator syntax. Do not add duplicate arithmetic function wrappers solely to
mirror NumPy. Current public functions are `fnp.sum(a, axis=None)`,
`fnp.transpose(a)`, `fnp.equal(a, b)`, `fnp.min(a, axis=None)`,
`fnp.max(a, axis=None)`, `fnp.unique(a)`, `fnp.ones_like(a)` and
`fnp.column_stack(tup)`, which delegate distributed inputs to the existing tensor
methods. The extrema and unique functions use their approved aggregation kinds.
Existing `.sum()`, `.T`, `==`, indexing and `len(x)` remain available. Concrete
NumPy arrays use the corresponding local NumPy computation; unsupported input
types raise TypeError without attempting to convert distributed arrays.

The sum overloads preserve the tensor method's scalar/full-reduction and
axis-dependent return types. Transpose returns `FedTensor` for a distributed
input and a NumPy array for a concrete input. Only the existing parameter subset
is public: axis for sum and default axis reversal for transpose. Custom transpose
axes and sum dtype/out/keepdims are not supported. Functions such as sqrt, clip, logical mask combinations and concatenation require separate operation implementations;
they are not enabled merely because NumPy provides them.

Pearson uses `fnp.sum` for feature and squared sums, and `fnp.transpose(y) @ x`
for cross-products. Its six aggregation steps, output orientation and ordinary
NumPy/SciPy postprocessing are unchanged. This interface addition does not change
RPC schemas, worker operations, deployment settings or privacy boundaries.

Algorithm authors import `FedTensor` from `exaflow.algorithms.feta`. Its abstract
interface declares arithmetic, equality, indexing, matrix multiplication,
transpose, sum and length
with typed operands and results. Prepared runtime tensors inherit that interface;
algorithm `run` parameters are annotated with it. Elementwise arithmetic and
transpose return `FedTensor`; length returns `int`. Overloads specify that
`sum()` and `sum(axis=None)` return a concrete `int` or `float`. Axis-specific
reductions retain union return types because partitioning and rank determine
whether their results are tensors, arrays, or scalars. Matrix products return
`FedTensor | NDArray`, since supported matrix operands are two-dimensional.

Production parity validation compares both algorithms through `/analysis` using
the same explicit complete-case filters. Normal Pearson retains its required
missing-value handler (a no-op after filtering); FETA uses no preprocessing.
Output structure,
labels and observation counts must match exactly; numerical outputs use
`rtol=1e-10`, `atol=1e-12` to allow summation-order rounding differences.

### Independent t-test Migration

`feta_ttest_independent` receives typed required x/y tensors. Text x retains its
labels, and y remains numerical. Column selection, equality and boolean indexing
prepare two stored local samples; six immediate reductions compute both counts,
sums and squared sums. Controller NumPy/SciPy preserves the existing pooled
variance formulas, output fields, errors and nonfinite edge behavior. No
preprocessing or automatic reduction batching is added. See the
[algorithm documentation](../algorithms/feta_ttest_independent.md).

### Global Unique, Minimum and Maximum / One-way ANOVA

Confirmed after discussion: category discovery uses `fnp.unique`, with each
worker computing distinct labels and the aggregation server returning their
sorted union as a NumPy array. `fnp.min` and `fnp.max` use global minimum/maximum
when contracting a partitioned axis, otherwise returning a local FedTensor.

Distributed unique initially accepts Unicode category-label tensors only.
Numerical global unique is not enabled as a way to release distinct outcome
observations. Concrete numerical NumPy arrays execute unique locally. The
initial API supports values-only unique (flattened input), and axis-only min/max;
return_counts/indices/inverse, custom dtype, out, keepdims and initial are not
exposed. Negative federated reduction axes remain unsupported.

A typed StringArray transport carries distinct labels and their Unicode dtype.
Workers report actual unique-result lengths; controller checks each against its
input-size bound and checks dtype, partition and lineage. Aggregation preparation
includes explicit kind SUM_VALUES/MIN_VALUES/MAX_VALUES/UNION_VALUES. Union
contribution lengths and widths are bounded by preparation, unknown kinds fail,
and malformed contributions terminate waiting fetches. Union memory reservation
uses conservative input-size bounds. Labels are sorted and deduplicated, including
labels absent on individual workers. Empty local sets contribute an empty array.

Min/max operate on real numerical distributed tensors. Empty worker subsets
contribute positive/negative infinity for float minima/maxima, or dtype bounds
for integers. Globally empty reductions raise the NumPy-style no-identity error.
NaNs propagate and real infinities are retained. Missing-aware extrema are not
implemented by these functions.

`feta_anova_oneway` is implemented with typed x/y inputs, observed-category
union, stored filtered samples, and `1 + 5*k` immediate aggregations for k groups.
Only eligible workers contribute categories. Mathematical inference is copied
into FETA-owned `AnovaStatistics`; result formatting is also FETA-owned. The
shared federated core and all Exareme3 implementations remain unchanged. FETA
does not use their execution scaffolding. Preserve response field
spelling, Tukey calculations, the reported standard-deviation formula,
mean-plus/minus-standard-deviation bounds and zero residual-variance behavior.
See the [ANOVA documentation](../algorithms/feta_anova_oneway.md).

Deploy FETA controller, workers and aggregation server together for the new
kind and string-array schema. Existing Exareme3 RPCs and deployment configuration
are unchanged. The label and per-group extrema releases match existing ANOVA;
privacy thresholds, lack of preprocessing and trust boundaries remain unchanged.
Rollback restores FETA code/bindings and removes
the new registration. No data migration is required.

### Enforced FETA Ownership Boundary

The following implementations are independent copies, maintained within FETA:

- Each `algorithms/feta/feta_*.py` owns its specification and response model;
  Pearson also owns its matrix formatter.
- `algorithms/feta/_anova.py` owns ANOVA inference and output formatting.
- `feta/inputdata.py` owns the worker/controller data-selection model.
- `worker/feta/data_loader.py` and `filters.py` own SQL loading and filter building.
- `feta/serialization.py` owns numeric Arrow encoding/decoding; operation contracts
  retain FETA's typed Unicode transport.

The dependency test `tests/standalone_tests/feta/test_dependency_boundary.py`
checks all production FETA modules, including the aggregation implementation,
and permits only FETA-owned modules and named public engine integration contracts.
No production imports from `algorithms/exareme3`, `worker/exareme3`,
`algorithms/federated`, `algorithms/utils`, shared filter helpers or shared
aggregation serialization remain. The earlier shared ANOVA-core modification
is reverted; its legacy source is unchanged.

Duplicated implementation changes must be maintained and validated explicitly
within each paradigm. Keep cross-paradigm parity tests as reference checks,
without creating production dependencies between them. This ownership change
preserves HTTP results, protobuf schemas, configuration, privacy and cleanup;
rollback restores only FETA implementations and imports, requiring no data migration.

### Two-way ANOVA / Design Assembly

`feta_anova_twoway` is implemented independently within FETA. Its owned
specification/result model and copied table inference do not import or change
Exareme3 or shared algorithm libraries. Selected x contains two text factors and
y one numerical outcome. Global observed levels use existing category union.

NumPy-equivalent `ones_like` and `column_stack` are added as approved local
operations. `ones_like` retains numerical/boolean dtype, shape and partitioning;
no dtype/shape override is exposed. `column_stack` accepts aligned row-partitioned
vectors/matrices, turns vectors into columns and returns a FedTensor. All
distributed inputs must be FedTensors in the same context/row layout. Empty input
lists, mismatched local row counts, column-partitioned inputs and mixed local/
distributed inputs fail. Mixed text/numerical columns are rejected to avoid
converting observations into labels for global unique. Allocation is bounded
before the NumPy kernel. Concrete NumPy arrays execute these functions locally.

Treatment indicators use equality times 1.0; interactions use multiplication.
Five designs are assembled explicitly. Preserve the reference's duplicate
intercept columns. Each model aggregates its Gram matrix and response
cross-product; controller NumPy pinv and matrix_rank compute coefficients/rank.
Local predictions use matrix multiplication by those coefficients, and squared
residuals are summed globally. Two label unions, one count and three sums per
model produce 18 immediate aggregation steps. Unused OLS metrics are omitted.

Preserve Type I factor ordering, Type II marginal differences, rank-based degrees
of freedom, nonnegative clipping and null/error behavior for degenerate tests.
See the [algorithm documentation](../algorithms/feta_anova_twoway.md).
The default sstype specification value retains the existing string "2" quirk;
request values remain validated as integers. Correcting the default is separate
from this migration.

The protobuf operation enum adds ONES_LIKE and COLUMN_STACK. Deploy FETA
controller and workers together; aggregation messages/kinds and existing
Exareme3 protocols are unchanged. No config/data migration or privacy-threshold
change is needed. Rollback removes the registration and design-operation
extensions and restores the prior FETA code/bindings. Legacy code is unchanged.

### FedColumns and Logistic Regression (2026-10-14)

The user selected a typed column collection to support mixed covariates while
preserving per-column metadata and dtypes. `FedColumns` exposes column selection
by name returning a FedTensor, ordered names, read-only metadata and global nrows
from shape metadata. It has no collection arithmetic, assignment, row filtering
or len semantics. Individual tensors share workers, context and local row layout;
construction and initialization validate those invariants.

`ColumnMetadata` includes original SQL type, categorical flag, declared codes and
display labels. Metadata is immutable. Column selection returns an existing
reference without dispatch. Worker initialization loads all selected columns in
one query and stores each homogeneous column independently. Integer collection
columns retain their native integer dtype; existing numerical matrix inputs
remain float64. Overlapping collection/matrix selections preserve each dtype.
No mixed object arrays or named-axis tensor metadata are introduced.

FETA algorithms declare `column_inputs` as static group-representation metadata;
there are no executable resolution hooks. The controller still selects x/y from
specifications and requests, loads only selected groups and supplies worker-
landscape column metadata. Omitted optional groups remain omitted. Logistic run
receives `x: FedColumns, y: FedTensor`; existing tensor algorithm signatures are
unchanged. Additional named input groups remain out of scope.

Typed protobuf initialization adds column_groups and named per-column responses.
The controller validates exact column/group presence, duplicates, shape, dtype,
partition and row alignment, including native integer descriptors. Controller
and workers must be updated together; HTTP requests/results are unchanged.

The next priority implementation is complete as `feta_logistic_regression`.
Categorical encoding is intrinsic model algebra, not a preprocessing pipeline.
Text levels use global string-label union. Numeric categorical levels use declared
codes and aggregated presence counts in numeric order; undeclared codes fail
explicitly. This does not enable global unique for numerical observation tensors.
Display labels are retained as metadata; coefficient names preserve code-based
reference naming. Categorical features precede numerical features, and one
intercept is added.

Approved NumPy-style exp/log/three-argument where are local operations. They
preserve partitioning, validate shapes/alignment and predict allocation. Where
requires a boolean tensor condition and numerical values; it cannot convert
observations to string labels. Custom dtype/out/where arguments and one-argument
where are not exposed. Ordinary NumPy arrays execute locally.

The Newton workflow aggregates count/positive count, then gradient, Hessian and
log-likelihood each iteration. Controller inv falls back to pinv only when inv
raises; convergence remains max(abs(gradient)) \<= 1e-4 with 50 iterations. Owned
summary helpers preserve standard errors, z-tests, 95% intervals, pseudo-R²,
likelihoods and AIC/BIC. Hessian/likelihood retain their pre-update iteration
semantics. Exact collinearity can make the reference inverse unstable and
individual coefficients sensitive to partition/kernel rounding; no solver-policy
change is introduced. See the [algorithm documentation](../algorithms/feta_logistic_regression.md).

All implementations are FETA-owned copies or new FETA code. Existing Exareme3
and shared algorithm libraries, data storage, privacy thresholds, cleanup and
configuration remain unchanged. Rollback removes the new registration and
restores FETA inputs/operation code and bindings; no data migration is involved.
Logistic cross-validation is not part of this priority migration. The remaining
statistics migration still resumes at chi-squared, after the user's checkpoint.

### Chi-squared migration (2026-10-14)

`feta_chi_squared` receives both selected groups as `FedColumns`, retaining
category metadata. Metadata preserves declared code order; algorithms requiring
sorted levels, including logistic regression, sort their levels explicitly.
Contingency counts use float indicator designs and `factor.T @ outcome` with
one automatic sum aggregation. No new local operation is required. Declared
categories are deduplicated; absent rows/columns are pruned. Empty metadata uses
sorted observed string union. Unknown codes outside a declared domain are
ignored, matching the reference contingency table. SciPy's default continuity
correction, output orientation and category-code labels are preserved. Empty
tables raise a FETA input error and contexts are cleaned up.

See [algorithm documentation](../algorithms/feta_chi_squared.md). Statistics
migration still proceeds one algorithm at a time; Fisher's exact test is next
only after user confirmation.
