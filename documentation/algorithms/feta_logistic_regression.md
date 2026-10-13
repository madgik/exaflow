# Logistic regression (`feta_logistic_regression`)

## Overview

Fit a binary logistic regression with an intercept and numerical and/or
categorical covariates. FETA owns its specifications, response models, numerical
workflow and inference helpers. The existing `logistic_regression` implementation
is used only as a validation reference.

## Inputs

Required `x` selects covariates and arrives as `FedColumns`. Required `y` selects
one nominal text outcome and arrives as a `FedTensor`. `positive_class` assigns
1 to the selected outcome code and 0 to every other outcome code.

The collection exposes `x[name]`, `x.names`, `x.metadata[name]` and `x.nrows`.
Each column is an aligned single-column FedTensor, retaining text labels or its
numerical dtype. Integer covariates preserve their stored integer dtype in
collection inputs; the final numerical design uses float64. Existing matrix
input algorithms continue to load numerical data as float64.

Column metadata includes original SQL type, categorical status, declared category
codes and display labels. Names preserve request order. Metadata and collection
properties are read-only; column selection returns existing references without
worker requests. There is no collection arithmetic, assignment, row filtering
or ambiguous len(collection) behavior.

Initialization loads all selected inputs in one query and skips workers below
the existing privacy threshold. Missing inputs are rejected and preprocessing
is unsupported. Intrinsic category encoding is part of the model calculation.

## Statistical Method

Categorical features use treatment coding: omit the first sorted observed level,
then create one indicator for each remaining level. Encoded categorical columns
precede numerical covariates, preserving the existing coefficient-name order.
The design begins with one intercept.

For design D and binary target t, start coefficients at zero and iterate:

```text
eta = D @ coefficients
mu = 1 / (1 + exp(-eta))
w = mu * (1 - mu)
gradient = D.T @ (t - mu)
hessian = D.T @ (w * D)
ll = sum(where(t == 1, log(mu), log(1 - mu)))
inverse = inv(hessian), falling back to pinv if inv raises LinAlgError
coefficients = coefficients + inverse @ gradient
stop when max(abs(gradient)) <= 1e-4
```

A maximum of 50 iterations preserves the original convergence error. Global
counts must exceed the number of design columns both overall and within each
binary outcome class, matching the reference checks and messages.

Standard errors derive from the diagonal of the final Hessian inverse. Normal
z-tests and 95% confidence intervals, model/residual degrees of freedom,
Cox–Snell/McFadden pseudo-R², null/fitted log-likelihood, AIC and BIC retain the
reference formulas. The returned coefficients include the final update; Hessian
inverse and likelihood belong to that iteration's pre-update coefficients,
preserving the existing stopping behavior.

## Computation Without Row-level Data

String categorical levels use global label union. Numeric categorical columns
use declared category codes and globally aggregated presence counts, preserving
numeric level ordering. A numeric category value outside its declared domain
produces a clear error. Global unique remains unavailable for numerical
observation arrays.

Only category labels, counts, gradients, Hessians and likelihood totals are
released. Covariate/outcome rows, encodings, probabilities and weighted designs
remain stored on workers.

### Federated Flow

```python
# Controller prepares x: FedColumns and y: FedTensor.
# Encode observed categories using comparisons, and select numerical columns.
t = (y == positive_class) * 1.0
D = fnp.column_stack([fnp.ones_like(t), *encoded_columns, *numerical_columns])
# Aggregate observation/positive counts and validate class sizes.
for iteration in range(50):
    eta = D @ coefficients
    mu = 1.0 / (1.0 + fnp.exp(0.0 - eta))
    gradient = D.T @ (t - mu)
    hessian = D.T @ ((mu * (1.0 - mu)) * D)
    ll = fnp.sum(fnp.where(t == 1.0, fnp.log(mu), fnp.log(1.0 - mu)))
    # Update coefficients with global NumPy matrices and test convergence.
# Format global inference and clean up context state.
```

Each iteration consumes three immediate aggregation steps. Initial counts use
two sums, with additional label-union or declared-category presence steps for
categorical covariates. There is no batching or automatic retry.

## Technical Decisions

The algorithm declares `column_inputs = frozenset({'x'})`; this is static input
representation metadata, not an executable input-resolution hook. The controller
still derives selected groups from the specification and request, passes only
selected groups to run, and supplies column metadata from the worker landscape.

Typed initialization messages identify column groups and return named per-column
references. Workers store homogeneous arrays individually, without object arrays.
Requested matrix groups continue to return their original descriptors. Overlapping
matrix/column selections retain each representation's dtype without conflation.

`fnp.exp`, `fnp.log` and three-argument `fnp.where` are fixed approved operations.
They retain distributed placement, validate row alignment and predict allocation.
Where requires a boolean tensor condition and numerical values. It cannot convert
numerical observations into strings. NumPy array inputs execute locally.

All mathematical helpers are copied into FETA. Public platform interfaces remain
integration contracts; Exareme3 and shared algorithm libraries are unchanged.

## Outputs

`dependent_var` is the outcome name. `indep_vars` lists Intercept, encoded
categorical feature names and numerical feature names. `summary` contains:

- n_obs, coefficients, stderr, lower_ci, upper_ci, z_scores and pvalues.
- df_model, df_resid, r_squared_cs and r_squared_mcf.
- ll0, ll, aic and bic.

Feature names retain category codes, matching the existing algorithm; display
labels remain available in column metadata.

## Validation

Federation simulations compare every output field with the normal algorithm
across numerical-only, categorical-only and mixed covariates, including numeric
category codes, uneven/empty partitions, privacy skips and Unicode labels.
Statsmodels Logit independently checks coefficients on identifiable designs.
Tests cover typed reference reuse, dtype preservation, read-only metadata,
unknown numeric codes, class-size errors, convergence failure, solver fallback,
operation placement and cleanup.

Package tests execute Quart/DuckDB/gRPC with mixed covariates. Development-tool
parity tests compare all five original logistic fixtures through /analysis on
identical complete-case filters, with rtol=1e-10 and atol=1e-12.

## Limitations

Separation, ill-conditioned or non-identifiable designs retain reference solver
limitations. An exactly collinear Hessian may produce an unstable inverse instead
of raising, so individual coefficients can depend on partition/kernel rounding.
The pseudoinverse path is tested explicitly when inversion fails; this migration
does not change the solver policy.

Intermediate tensors are retained until context cleanup and constrained by the
existing memory/lifetime limits. Numeric categorical domains must be declared;
there is no unrestricted numerical category discovery or observation retrieval.
No preprocessing, regularization, cross-validation or convergence diagnostics
are added to the response. Privacy thresholds and cleanup remain unchanged.
