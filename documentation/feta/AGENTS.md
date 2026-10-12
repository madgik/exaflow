# FETA Architecture and Agent Guidance

Initial decision record: **2026-10-13**.

This is the canonical, evolving architectural record for FETA across algorithms,
controller, and workers. Read it alongside the repository's
[root agent instructions](../../AGENTS.md) before working on FETA. Keep FETA's
architectural documentation here; other repository guidance should link here
rather than duplicate these decisions.

This document records design decisions, not an implemented runtime. Keep
confirmed decisions, implementation proposals, and unresolved questions
separate. Update the record as discussion continues; do not silently promote a
proposal to a confirmed decision.

## Confirmed Decisions

### Identity, Ownership, and Execution

- The paradigm is named **FETA**. Use `feta` for its namespaces rather than
  naming it Exareme4. Exact module layout remains open.
- FETA has a separate execution flow and orchestration, while continuing to
  use the existing aggregation server. Existing Exareme3 helpers may be called
  directly where needed; this reuse does not route execution through Exareme3.
- Do not move code out of Exareme3 to make it reusable by FETA. Use existing
  helpers in their current locations, including the worker data loader.
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
  names fail explicitly. Exact module layout and discovery configuration remain
  open.

### Context Initialization and Algorithm Inputs

- The runtime prepares the initial `x` and `y` tensors before invoking the
  algorithm and passes them directly to `run(self, x, y)`. They are already
  `FedTensor` objects; the algorithm does not cast them or select input columns
  through a `self.tensor(...)` call.

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

- Load initial numerical tensors as `float64`, matching the current Pearson
  computation. This does not change the integer return type of `len(x)` or
  settle the complete set of supported literal operand dtypes.

- For `feta_pearson_correlation`, resolve omitted `x` to the ordered `y`
  columns, preserving existing Pearson behavior. When the ordered selections
  are identical, both inputs may reference the same stored tensor. This is a
  Pearson input rule, not a universal fallback for future algorithms.

Initialization proceeds as follows:

1. The controller resolves ordered input columns and candidate workers.
1. It sends an `InitializeContext` lifecycle request with the context ID,
   validated data selection, and input column groups.
1. Each worker loads the required columns once, in one DuckDB query applying
   dataset selection and filters. Reuse the existing
   [Exareme3 data loader](../../exaflow/worker/exareme3/udf/udf_db.py) in place;
   do not extract it into another module.
1. The worker checks the privacy threshold and the temporary missing-value
   policy, then constructs `x` and `y` from the same loaded rows, restoring
   each group's requested column order.
1. The worker stores the prepared arrays and returns references and metadata;
   workers with insufficient data are skipped under the agreed policy.
1. The controller creates `FedTensor` objects for the eligible worker set and
   invokes `run(x, y)`.

The initialization flow is confirmed. Exact protobuf fields and lifecycle
response/error schemas remain to be specified.

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
  concurrent. Per-context lifecycle coordination with in-flight operations
  remains to be specified rather than introducing blanket tensor locks.
- Workers generate IDs for new results. Requests do not supply an output ID.
- Reuse stored intermediate results. Detailed caching policy remains open.
- At execution end, clean up each worker's context and all its associated
  values, including failure paths. Also clean up the aggregation-server
  context. Failure and cancellation mechanisms remain to be designed.
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
  The background scan interval and precise deadline-enforcement mechanics
  remain to be specified.
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
  context. The cancellation/error-propagation mechanism remains open.
- Automatic expiry for abandoned contexts is confirmed through the background
  cleanup approach above. It has not yet been implemented.

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
each individual aggregation. The setup, submission, and fetch message schemas,
result-release lifecycle, and timeout/cancellation behavior remain to be
designed. All three stages identify the aggregation by the shared context ID
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
is confirmed; exact message schemas and enum values remain to be specified.
Reuse the existing Arrow tensor byte encoding in
[aggregation_serialization.py](../../exaflow/aggregation_serialization.py)
for numeric arrays carried in protobuf byte fields. Worker-side semantic
validation is still required.

Use one `LocalOperation` class for every operation, containing:

- An operation type.
- Inputs consisting of local tensor references, supported ordinary Python
  values, or numeric NumPy arrays.
- Operation-specific parameters, such as a numeric axis.

Illustrative interface; the exact schema and enum names remain open:

```python
LocalOperation(
    type=OperationType.MULTIPLY,
    inputs=[LocalTensorRef("tensor-a81f"), 2.0],
    parameters={},
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
Local computations still use approved `LocalOperation` instances. How the
submission instruction references a stored partial result or accompanies a
local computation remains a protocol-design question.

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
  therefore requires aggregation-server protocol and lifecycle changes; it
  cannot be implemented solely by renaming existing client calls.
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

## Implementation Proposals — Not Yet Finalized

- Validate operation-specific fields on the worker, independently of controller
  validation. Proposed checks include supported operation types and argument
  schemas, context-scoped references, actual stored shapes and numeric dtypes,
  valid axes, and resource limits.
- Bound operation allocations and stored intermediates. Limits and enforcement
  mechanisms remain open.
- Expose separate aggregation-server prepare, contribution-submission, and
  result-fetch calls to implement the confirmed controller-driven flow. Exact
  RPC names and schemas are not finalized. Document the impact on existing
  Exareme3 consumers when implementing changes to the shared server.

Do not interpret these proposals as an implemented security mechanism or a
finalized protocol.

## Remaining Questions

- Exact initialization protobuf and lifecycle response/error schemas. The
  loading flow, prepared `float64` input tensors, `x`/`y`-only initial scope,
  Pearson's omitted-`x` behavior, and insufficient-data skipping are settled.
- Exact module layout and discovery configuration. A FETA-specific base class,
  module discovery, name-keyed registry, and shared API routing are settled.
- Final protobuf schemas, exact supported literal dtypes, and operation set.
  Protobuf transport, Arrow tensor byte encoding, and sending ordinary NumPy
  array inputs to every worker are confirmed.
- Further helper reuse as implementation needs arise. Existing Exareme3 code
  stays in place and may be called directly; FETA orchestration stays separate.
- Aggregation setup, submission, and result-fetch schemas; submission
  acknowledgements; result release after controller retrieval; and how
  worker errors, deadlines, or cancellation terminate a waiting fetch.
- Store concurrency, resource limits, caching, cancellation, and cleanup after
  worker failure. Failing the execution after a worker failure and disabling
  automatic operation retries are confirmed.
- Configuration wiring, sweep interval, precise deadline enforcement, and
  in-flight-operation lifecycle for maximum-lifetime expiry. The initial
  default is 60 minutes; heartbeat renewal is out of scope.
- Caller authorization and permitted result-release policies.
- Tensor behavior beyond the agreed initial row-partitioned examples,
  including broader broadcasting and length semantics after transformations.

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
- Moving Exareme3 helpers into neutral modules for FETA reuse. Call the
  existing helpers in place instead.

## Documentation Maintenance

Keep confirmed decisions consistent with the discussion and mark proposals as
such. Verify links when moving or extending this record. Runtime, public API,
protobuf, dependency, and deployment changes require their own implementation
work and validation; this documentation record does not itself make those
changes.
