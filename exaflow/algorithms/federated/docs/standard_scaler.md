# FederatedStandardScaler

## Interface and fitted lifecycle

Import `FederatedStandardScaler` from `exaflow.algorithms.federated.feature_engineering`
or `exaflow.algorithms.federated`. It implements the existing
`FederatedTransformer` named-DataFrame interface through `FeatureTransformer`.

```python
scaler = FederatedStandardScaler()
scaler.fit(
    agg_client=agg_client,
    data=training_frame,
    categorical_vars=[],
    numerical_vars=["crp"],
)
values = scaler.transform(new_frame, categorical_vars=[], numerical_vars=["crp"])
```

`fit()` returns `None`, consistent with the native transformer interface.
`values` is a two-dimensional NumPy array. Fitting stores `mean_`, `scale_`,
and `feature_names_in_` only after successful validation. A successful refit
replaces that state; a failed refit does not publish a partially updated schema.
There is no `recipe_`, `from_recipe()`, serialization model, or persistence API.

## Numerical method and aggregation

For each feature, aggregate its observed sum and count to obtain the mean.
In a second pass, aggregate squared deviations `S` and residual deviations `D`.
Population variance is `max((S - D**2 / n) / n, 0)`.
As before, the scale is one when variance is below the numerical error bound
`n * eps * variance + (n * mean * eps)**2`; otherwise it is the square root of
variance. The fit retains the original three sum-aggregation calls and does not
send raw observations. Shared numerical helpers are owned by
`feature_engineering/numerical_utils.py`; remaining F1 primitives import those same
helpers rather than maintaining another implementation.

Missing values are ignored during fitting and preserved by transformation.
Empty local partitions contribute neutral sums and counts; their transforms
have shape `(0, n_features)`. Infinite inputs and globally empty or all-missing
features remain invalid. Participants must supply aligned feature layouts; the
existing caller is responsible for distributed error propagation and cleanup.

Neither the mathematical class nor its runtime step imports deployment privacy
settings. The existing worker row checks and downstream report masking are
unchanged; the scaler has no additional per-dataset or non-missing count floor.
All observed values enter the fit rather than silently excluding small groups.
Report masking does not remove their influence on learned parameters. Stronger
composition guarantees require a separately reviewed policy.

## Validation and composition

The transformer requires a DataFrame and distinct numerical column names.
Transform requires the same selected names/order as fit. Fitted means must be
finite, and scales must be positive, finite, and feature-aligned. Transform does
not aggregate, refit, mutate the caller's frame, or modify learned parameters.

Tests compare results with scikit-learn 1.7 StandardScaler, exercise native
single-transformer `FederatedPipeline` and `FederatedColumnTransformer` usage,
and retain the previous numerical regressions. Neither container is changed.
The common Protocol and this concrete transformer's fit both return `None`.
The runtime wrapper fits all ordered `source_variables` in one vectorized
operation and appends their corresponding `new_column_names`. Changing the
feature count changes aggregate-vector widths, not the three aggregation calls.

The public preprocessing wrapper, contract change, and limitations are
documented in [Standard Scaler](../../../../documentation/algorithms/standard_scaler.md).
