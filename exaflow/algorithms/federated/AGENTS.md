# Federated Library Development

This file is the canonical development context for this library. Treat the
package as an independent statistics library: algorithms describe mathematical
operations and library contracts; applications consuming it own presentation,
request handling, transport, and execution lifecycle.

Method documentation lives in [docs](docs/). The [README](README.md) indexes the
available methods and components. Keep implementation details and method
explanations there instead of duplicating them in application documentation.

## Scope and dependency boundary

- Import from this package, the standard library, and third-party libraries.
  Package-qualified self-imports use `exaflow.algorithms.federated` in the current
  checkout. Do not import other modules from the enclosing application package,
  including inside functions or type-checking blocks.
- Callers supply data, grouping-column names, thresholds, configuration, and an
  aggregation client. Do not obtain these from application constants, global
  service configuration, environment variables, or service registries.
- Keep the mathematical core usable with in-memory arrays/dataframes and
  simulated aggregation clients. Creating or renaming an independently
  installable distribution is a separate packaging concern.
- Callers coordinate participation, cancellation, propagation of failures, and
  resource cleanup. Library checks raise locally and do not implement a service
  lifecycle or a transport protocol.
- Preserve existing threshold checks and suppression behavior. Exact sufficient
  statistics and fitted results are not a guarantee that every returned value
  is suitable for release; consumers choose their result-release policy.

## Architecture and interfaces

- Organize implementations by statistical domain: `statistics`, `linear_model`,
  `decomposition`, `naive_bayes`, `mixed_effects`, and `cluster`.
- Encoders belong in `feature_engineering`; column composition and passthrough
  in `compose`; outlier reporting and clipping helpers in `outliers`.
- Shared splitting, cross-validation, and scoring infrastructure belongs in
  `model_selection/cross_validation`. Mixed-effects numerical helpers live in
  `mixed_effects/mixed_effects_utils.py`.
- Reuse the protocols in `utils/interfaces.py` and `utils/agg_client.py`.
  Preserve each component's existing fitting, transformation, prediction, and
  results contract rather than imposing a new uniform estimator API.
- The aggregation-client protocol supplies `sum`, `min`, `max`, and `union`.
  Implementations must agree on operation order, vector shapes, category order,
  and model configuration across participants. Local checks must not introduce
  unmatched collectives or change the statistical population being aggregated.
- Prefer existing family utilities, validation helpers, composition primitives,
  and numerical routines over duplicated implementations.

## Numerical and validation conventions

- Python targets 3.11. Match the existing 88-character line length, double
  quotes, and single-line imports. Leave formatting and import sorting to
  commit hooks and CI; do not run Ruff manually.
- Use public NumPy, SciPy, pandas, scikit-learn, and statsmodels APIs where
  appropriate. Do not introduce estimator inheritance merely to reuse a public
  validation function.
- Shared checks live in `validation`; their API and method coverage are
  documented in [input validation](docs/input_validation.md).
- Apply validation to effective inputs after the method's existing conversions.
  Preserve supported integer dtypes, categorical ordering, missing-value
  handling, reshaping, and numerical fallbacks.
- Do not impose blanket restrictions on empty local partitions, constant data,
  zero-feature matrices, local class availability, or finite values where a
  method intentionally supports them. Enforce the method's actual prerequisites.
- Separate local input structure from global statistical requirements. Use
  existing aggregate counts for sample requirements; do not add validation
  rounds without a demonstrated need.
- Keep successful numerical results and existing collective ordering stable
  unless the requested change explicitly alters the method. Explain any changes
  to numerical behavior, assumptions, defaults, or result fields.
- Do not add legacy aliases, silent normalization, or compatibility paths unless
  explicitly requested.

## Exceptions and diagnostics

Define library-authored failures with `FedValueError`, `FedTypeError`,
`FedStateError`, or `FedObservationError` from [exceptions.py](exceptions.py).
These preserve the corresponding Python exception semantics:

| Type | Condition |
| --- | --- |
| `FedValueError` | Unsupported argument values or statistical prerequisites. |
| `FedTypeError` | Unsupported argument or component types. |
| `FedStateError` | An operation is invalid in the current component state. |
| `FedObservationError` | Observations do not meet a method-specific requirement. |

Every explicit failure raise must supply a `FedErrorReason` member. Define
reasons in `exceptions.py` using `error_reason(template)`. Stable string values
derive from the lowercase member names; identical diagnostic templates do not
make two reasons interchangeable. Reuse a reason only for the same condition.

```python
from exaflow.algorithms.federated.exceptions import FedErrorReason
from exaflow.algorithms.federated.exceptions import FedObservationError

raise FedObservationError(
    FedErrorReason.INSUFFICIENT_FOLD_SAMPLES,
    n_samples=n_rows,
    n_splits=n_splits,
)
```

The constructor enforces the enum type, stores `.reason` and `.details`, and
formats the template into `.message` and `str(error)`. Missing required arguments
or template fields retain Python's `TypeError` and `KeyError` behavior.

Describe the library condition without assigning application responsibility.
Do not encode presentation categories, request terminology, or instructions to
adjust a user interface. Do not include raw observations or sensitive values.

Use distinct reasons for invalid input and invalid component output or computed
state. When adding context at a component boundary, catch the specific failure
and raise the appropriate library error with `from exc`. Keep the protected
operation narrow; preserve unrelated failures. Bare re-raises are permitted.
Unexpected dependency exceptions retain their normal behavior; do not wrap every
exception as a declared library failure.

## Method documentation

- Keep one canonical method description in `docs/`, beside the library code.
  Document shared encoders, splitters, scorers, and composition components once
  and link to them from methods that use them.
- Describe purpose, library inputs and defaults, result objects, mathematical
  model/formulas, aggregated quantities, `### Federated flow` pseudocode,
  numerical decisions, errors, reference comparisons, and limitations as relevant.
- Read the implementation and focused tests before changing a method page.
  Distinguish exact calculations from approximations and actual defaults from
  reference-package defaults. Do not claim equivalence that the implementation
  or comparisons do not establish.
- Use library terminology: arrays, features, outcomes, groups, participants,
  aggregation clients, and results. Do not describe application request fields,
  response models, display labels, transport status codes, or external adapters.
- Keep documentation links within the library for library contracts. Consumer
  documentation can link into this library without becoming part of its API.
- Update the README index when introducing a method or component page.

## Focused checks

Tests currently live in the checkout's test tree. Run commands from the checkout
root; service deployment is not required for library-only work.

```bash
uv run pytest -q tests/standalone_tests/federated_algorithms/<family>/test_<algorithm_id>.py
uv run pytest -q tests/standalone_tests/federated_validation
uv run pytest -q tests/standalone_tests/test_federated_import_boundary.py tests/standalone_tests/test_federated_exception_convention.py
```

Use the shared helpers in `tests/standalone_tests/federated_algorithms/utils`
for centralized comparisons and simulated multi-participant aggregation. Choose
checks for the changed behavior: reference parity, uneven or empty partitions,
missing/constant data, category alignment, numerical limits, or error contracts.
Do not change expected numerical fixtures merely to accommodate a failure.

The import-boundary check rejects imports outside this library in the enclosing
package. The exception-convention check requires explicit library failures and
reasons; constructors enforce enum membership. Consumer-specific checks are owned
by the consumers.

## Library-only workflow and completion

1. Read the relevant implementation, method page, protocols, and focused tests.
1. Implement the mathematical/component change in its domain and update exports
   when adding public components.
1. Update the canonical library method/API documentation and focused coverage.
1. Run relevant library checks and report any unrun checks with their reason.

Library-only work does not require application wrappers, request specifications,
deployment artifacts, or consumer integration scaffolding. When integration is
also requested, its owner completes the consuming application's workflow.

Completed work has no placeholders, preserves the dependency boundary, uses
explicit diagnostic reasons, documents its actual library contract, and passes
the relevant checks. Report numerical or API changes and their practical limits.
