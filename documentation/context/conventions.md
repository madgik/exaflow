# Conventions

## Confirmed Conventions

### Python and Formatting

- Python target is 3.11.
- Line length is 88.
- Formatting uses double quotes and spaces.
- Imports are organized as single-line imports.
- Agents must not run Ruff manually; commit hooks and CI own these checks.

### Package Management

- Root Exaflow package uses uv.
- `exadata-validator` is a separate uv-managed package with its own lockfile.
- CI installs dependencies with uv and often uses `--no-install-project` for root
  Exaflow tests.

### Configuration

- Controller config is loaded in `exaflow/controller/__init__.py`.
- Worker config is loaded and normalized in `exaflow/worker/__init__.py`.
- Aggregation server config is loaded in `exaflow/aggregation_server/__init__.py`.
- Local deployment config starts from `.deployment.sample.toml` and generates
  files under `configs/`.
- Algorithm discovery uses `EXAREME3_ALGORITHM_FOLDERS`.

### API and DTOs

- Analysis validation errors call selected inputs variables. CDE refers to
  metadata definitions; column refers to tabular operations. This terminology
  changes error text only, not validation rules, status codes, or schemas.

- FastAPI endpoint handlers live in `exaflow/controller/http/endpoints.py`.

- Request DTOs and validators live under `exaflow/controller/services/api`.

- Public analysis form shape is defined by common inputdata, preprocessing, and
  algorithm specifications exposed under `GET /specifications/*`.

- `documentation/api-specification.md` documents the API contract for clients.

### Algorithms

- Runtime Exareme3 algorithm wrappers live under
  `exaflow/algorithms/exareme3`.
- Group runtime reports, preprocessing steps, and model CV variants by domain.
  Do not create a generic preprocessing folder. Base classes and recursive
  discovery determine registry membership. Keep validation and result contracts
  in the module that owns the behavior; related modules may import them directly.
  Extract a helper module only for a demonstrated need. Keep shared metadata
  transforms in `utils/metadata.py`.
  Runtime domain and federated family are separate tooling inputs:
  `--runtime-domain` controls wrappers, `--family` controls core placement.
- Federated algorithm core logic lives under `exaflow/algorithms/federated`,
  grouped by domain: `outliers`, `feature_engineering`, and `compose` own outlier
  methods, encoders, and passthrough respectively. Keep shared cross-validation
  infrastructure in `model_selection/cross_validation`. Existing mixed-effects
  numerical helpers live in `mixed_effects/mixed_effects_utils.py`.
- Shared core input checks live in `federated/validation`; coverage and error
  conventions are documented in `federated/docs/input_validation.md`. Retain each
  method's existing conversions and
  apply explicit array/parameter policies rather than blanket restrictions.
  Validators raise `FedError` subclasses locally without aggregation. The caller owns
  distributed error propagation, cancellation, and cleanup. Exareme3 owns UI
  message translation through `utils/error_translation.py`. Keep successful
  result contracts unchanged
  and add validation tests under
  `tests/standalone_tests/federated_validation` without editing existing tests.
  Array checks default to float conversion; preserve an already-normalized
  integer outcome dtype when validation would otherwise discard a float copy.
  Reuse the categorical dtype returned by category-order validation when
  constructing pandas categorical values, while validating current categories
  on every transformation.
- Follow the [federated error contract](#federated-error-contract) when adding
  library failures and Exareme3 translation policies.
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
- Preprocessing owns the dataframe loaded for each UDF invocation and may modify
  or return it directly. Callers must use the returned dataframe; direct callers
  needing an unchanged input must copy it first. These mutations do not write
  back to DuckDB. Algorithm-internal copies may still protect reused inputs.
- Preprocessing steps that need aggregation declare
  `ComponentType.AGGREGATION_SERVER` in their specification components and must
  explicitly declare the `agg_client` parameter on `transform_data()`.
  Aggregation-aware preprocessing steps
  should use the base `transform_data_and_metadata()` implementation instead of
  overriding it. Do not rely on `**kwargs` for aggregation-client injection.
- New algorithm work should use `.agents/skills/exaflow-algorithm-scaffold` and
  `.agents/skills/exaflow-algorithm-validate`.

### Federated error contract

Define library-authored failures using `FedValueError`, `FedTypeError`,
`FedStateError`, or `FedObservationError` from
`exaflow/algorithms/federated/exceptions.py`. That module also owns
`FedErrorReason`, its unique stable codes, and diagnostic templates. Reuse an
existing reason for the same condition, or add a new member there. Describe the
library condition without assigning UI responsibility.

The shared constructor requires an enum reason and its diagnostic fields:

```python
from exaflow.algorithms.federated.exceptions import FedErrorReason
from exaflow.algorithms.federated.exceptions import FedObservationError

raise FedObservationError(
    FedErrorReason.INSUFFICIENT_FOLD_SAMPLES,
    n_samples=n_rows,
    n_splits=n_splits,
)
```

Invalid reason types raise `FedTypeError`; missing arguments and template fields
retain Python's `TypeError` and `KeyError` behaviour. Explicit failure raises
must supply a reason; bare re-raises are permitted. Unexpected dependency errors
and framework wrappers retain their normal handling.

Register every reason explicitly in the existing `_TRANSLATIONS` mapping in
`exaflow/algorithms/exareme3/utils/error_translation.py` as
`(exception_type, message_or_KEEP_MESSAGE)`. For example, the mapping includes:

```python
_TRANSLATIONS = {
    FedErrorReason.INSUFFICIENT_FOLD_SAMPLES: (
        BadUserInput,
        "There are too few usable observations for the requested number of "
        "folds. Reduce the number of folds or adjust the filters.",
    ),
    FedErrorReason.LOGISTIC_CONVERGENCE_FAILED: (
        AnalysisExecutionError,
        "Logistic regression could not converge to a stable solution.",
    ),
    FedErrorReason.MODEL_NOT_FITTED: (RuntimeError, KEEP_MESSAGE),
}
```

Choose the application policy by the actual condition and the responsibility for
correcting it:

| Category | Rule | Examples | Client behavior |
| --- | --- | --- | --- |
| `BadUserInput` | A known analysis requirement is violated and the user can make a meaningful adjustment to inputs, filters, preprocessing, or parameters. | Fisher requires two observed categories; too few observations; incompatible variable selections. | HTTP 460 or RabbitMQ `BadUserInput`, with actionable guidance. |
| `AnalysisExecutionError` | A recognized computational failure prevents a valid result without establishing that the user supplied incorrect input. | Logistic regression does not converge; K-means produces nonfinite centers from finite inputs. | HTTP 462 or RabbitMQ `AnalysisExecutionError`, with a safe explanation and no assignment of user fault. |
| System error | A programming, integration, infrastructure, or unexpected failure occurred. Use a native exception target for library contract failures. | Invalid internal configuration, nonnumeric transformer output, mismatched internal shapes, or a PCA result with the wrong row count. | Existing system-error responses with generic client text and internal diagnostics. |

Specifications own exposed parameter types, allowed choices, individual bounds,
and variable metadata constraints. If a core check rejects the same constraint
after request validation, treat it as an integration defect. Keep the core checks
for standalone library callers. Cross-parameter requirements and requirements
that depend on actual observations may still require runtime validation; for
example, `k_min <= k_max`, observed category counts, or finite numerical data.
Passing request validation does not guarantee numerical convergence or successful
execution.

Separate conditions that need different application policies with distinct
library reasons. In particular, invalid input observations and invalid computed
centers must not share a reason; transformer output is an internal contract rather
than user-supplied data. Keep these distinctions computational in the library,
without adding UI categories to `FedError` or its reasons. Do not infer category
from the source exception class, the phase of execution, or message text.

The reason selects both the application exception and message. `KEEP_MESSAGE`
preserves the diagnostic verbatim for any target exception. For either visible
category, use it only when the library text is suitable for clients; otherwise
register a safe replacement. Never expose raw data or sensitive diagnostics.
Runtime-owned request validation can raise `BadUserInput` directly. Unexpected
exceptions retain system-error handling; do not catch them broadly and relabel
them as `AnalysisExecutionError`.

Privacy checks retain `InsufficientDataError`, including the observation-threshold
reason's translation and HTTP 461, because it can skip a worker. Neither visible
analysis category skips workers or turns a failed analysis into a warning result.

Translation is handled by the shared worker UDF service and controller strategy.
They catch `FedError` and use `raise translate_federated_error(exc) from exc`,
preserving cleanup. Algorithm wrappers generally should not add their own
translation catches. Application exceptions live in `worker_communication`.
Worker gRPC carries `BadUserInput` as `INVALID_ARGUMENT`, `InsufficientDataError`
as `FAILED_PRECONDITION`, and `AnalysisExecutionError` as `ABORTED`; the controller
worker client reconstructs them. Other worker failures use `INTERNAL` and become
generic controller failures. Analysis failures preserve fan-out cancellation and
aggregation cleanup.

HTTP 462 and RabbitMQ `AnalysisExecutionError` extend the public error contract.
Deploy controller and worker changes together so the worker category is preserved.
Rollback reverts both ends of the routing change; no data migration is needed.

Both validator tiers run these existing checks once per invocation with selected
algorithms and report the result as `federated_error_contract`:

```bash
uv run pytest -q \
  tests/standalone_tests/test_federated_exception_convention.py \
  tests/standalone_tests/exareme3/test_error_translation.py
```

The checks enforce explicit failure types/reasons and complete translation
coverage. Algorithm integration is complete only when this check and the
algorithm-specific checks pass.

### Testing

- Fast tests live under `tests/standalone_tests`.
- Federated algorithm parity tests live under
  `tests/standalone_tests/federated_algorithms`.
- Prod fixtures live under `tests/prod_env_tests/expected`.
- Pytest markers include `slow` and `very_slow`.

### Deployment

- Dockerfiles exist for controller, worker, and aggregation server.
- Helm chart lives under `kubernetes`.
- kind/Helm are used by prod environment tests.

## Recommended Conventions To Verify

- Keep business/runtime orchestration out of FastAPI route handlers where possible;
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
  `FedError` subclasses, `BadUserInput`, `AnalysisExecutionError`, and validation
  errors. Follow the [federated error contract](#federated-error-contract) for
  classification and translation.
- Preserve detailed logs for operators, but do not log secrets or sensitive data.
- Do not hide controller/worker/aggregation failures behind generic success
  responses.

## Dependency Rules

- Do not add production dependencies without explicit justification.
- Keep root and `exadata-validator` dependency changes separate.
- Update lockfiles intentionally when dependency manifests change.

## Declared preprocessing outputs

Preprocessing creators use an optional `outputs` container with
`new_column_names_param` and an ordered `columns` list. A definition produces
one column unless `column_count_param` names a required, single-valued integer
parameter. Dynamic definitions repeat by its supplied positive count. Fixed and
dynamic definitions can appear together; names from the required text-list
parameter match the expanded output order, with exactly one name per generated
column. Count parameters may declare `min` for the UI but need not do so; runtime
resolution always requires positive counts. No literal `column_count` field is
supported. Optional output fields and enumeration-source references serialize as
`null` when unset, using standard Pydantic serialization without omission-only
serializers. Null `column_count_param` means one column; null `enumerations` is
valid only for numerical columns.
Each column has `type`/`stattype`; nominal columns require enumeration instructions.
`DICT_KEYS` reads a required dictionary parameter and optionally a default
category parameter. `PREFIXED_RANGE` reads a positive count and required prefix
parameter and generates categories starting at 1. There is no nested source path.
Shared resolution and validation live in
`exaflow/algorithms/utils/output_specifications.py` and are used by controller
validation and worker preprocessing hooks. Protobuf Struct represents numeric
parameters as doubles, so output-count and enumeration-range resolution accept
exact integer-valued floats on the worker as well as integers. Fractional, nonfinite, boolean, and
nonpositive counts remain invalid; public request parameter validation still
requires the declared integer type. The former single-column output enum
and scalar `code` creator requests are removed. See the API specification for
consumer migration details.

K-means replay carries its fitted prefix; the parameter builder supplies
actual fitted `k` and that prefix alongside the complete replay payload. Cluster
identifiers are one-based, use `enumeration_prefix` (default `cluster_`), and must
agree across reports, replay centers, and created columns. Replay validates the
current model directly, without a schema-version field or compatibility paths.
Service and consumer rollback must be coordinated.
