# Federated feature transformations

## Domain and interface

Scalers, imputers and nonlinear transforms live in `feature_engineering/`;
`Winsorizer` and `OutlierDetector` live in `outliers/`. The existing one-hot and
ordinal encoders remain canonical and unchanged. Each new transformer implements
`FederatedTransformer` through its shared `FeatureTransformer` base, not an
adapter. Its own methods perform the fit and transform.

```python
from dataclasses import asdict
from exaflow.algorithms.federated.feature_engineering import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(data=frame, categorical_vars=[], numerical_vars=["crp"], agg_client=client)
values = scaler.transform(frame, categorical_vars=[], numerical_vars=["crp"])
state = asdict(scaler.recipe_)
replay = MinMaxScaler.from_recipe(state)
```

Fit returns `None`, transform returns a two-dimensional ndarray, and
`get_feature_names_out` returns the selected names in fitted order. Inputs are
DataFrames; unselected columns are not read or changed. All participants must
use identical ordered feature names and settings. The existing pipeline and
column-transformer composition rules are unchanged; this work does not add
arbitrary intermediate-array chaining to `FederatedPipeline`.

## Mathematics and aggregated quantities

| Primitive | Method | Shared quantities |
| --- | --- | --- |
| `SimpleImputer` | Replace missing values by mean, approximate median, constant, or approved categorical mode. Numeric constant defaults to zero. | Sums/counts; existing percentile rounds; or counts indexed by approved vocabulary. No observed-value union. |
| `MissingIndicator` | One for missing and zero for observed. | None. |
| `Log1pTransformer` | `log(1+x)`, with the explicit domain `x >= 0`. | A global domain-error count during fit. |
| `SqrtTransformer` | `sqrt(x)`, `x >= 0`. | A global domain-error count during fit. |
| `AsinhTransformer` | `asinh(x)` for real input. | None. |
| `StandardScaler` | `(x-mean)/scale`, population scale. Corrected two-pass variance avoids subtracting nearly equal raw moments. | Sum, count, centred squared sum and centred correction. |
| `RobustScalerApprox` | `(x-median)/(Q3-Q1)`. | Existing aggregated percentile estimates. |
| `MinMaxScaler` | `(x-min)*(b-a)/(max-min)+a`. Constant fitted ranges use denominator one. | Global minima and maxima. |
| `MaxAbsScaler` | Divide by global maximum absolute value. | Maxima and valid counts. |
| `Winsorizer` | Clip to fitted lower/upper pooled quantiles. | Existing aggregated percentile estimates. |
| `OutlierDetector` | Flag absolute population z-score, or median distance divided by IQR, exceeding threshold. Missing values have flag false. | Two-pass moments or shared quartiles. |
| `YeoJohnsonTransformer` | Maximise the Yeo-Johnson profile likelihood per feature, including its Jacobian. | Global extrema, count/Jacobian, then two corrected variance exchanges per evaluated candidate. |
| `QuantileTransformerApprox` | Interpolate approximate global ranks; optionally apply inverse standard-normal CDF, clipping ranks to `[1e-7, 1-1e-7]`. | Counts, extrema, packed histogram refinements and supported-group extrema. |

Missing values remain missing except for imputation and diagnostic masks.
Degenerate standard/robust/max-abs scales use one. No primitive mutates its input
DataFrame. Numerical errors and invalid fitted states raise `BadInputError`.

### Federated flow

```text
select the same ordered columns at every participant
validate the method's local domain and shared settings where required
exchange the method's aggregated sufficient quantities
derive common fitted parameters
apply those parameters to local values only
```

Independent callers own cancellation, error propagation and aggregation cleanup.
Shared Yeo-Johnson/quantile decisions must depend on global values, not local
constant/missing patterns, so every participant sends matching aggregation rounds.

## Approximation and privacy limitations

Robust scaling, winsorization and median imputation reuse the existing percentile
implementation and its approximation; they are not exact order-statistic fits.
Quantile transformation uses a first equal-width histogram followed by at most
`max_refinement_rounds` refinements (default two). All target groups in a round
share one packed count exchange. Selected bins are expanded to the effective
global support floor before requesting their extrema. Each fitted quantile is
recorded inside its achieved value interval. Minimum support or the round budget
can leave a wide interval; no distribution-independent small relative error is
claimed. Message count is bounded per feature/round, while payload and local work
grow with selected groups and bin count.

The effective support floor is `max(platform_minimum, configured_minimum)`, using
the maximum settings across participants. The runtime obtains platform policy
from the existing worker settings, not from caller input. Exact global endpoints
are still retained. Counts/extrema reveal aggregate information: these guards are
not differential privacy and do not guarantee resistance to repeated-query attacks.

Yeo-Johnson search starts near `[-5, 5]`, expands within float64-safe bounds, and
uses at most the configured evaluations. Diagnostics distinguish convergence,
constant features and approximation due to budget, numeric bounds or a flat
objective. A converged local search is not a certificate of a global optimum.

## Fitted state and compatibility

Fitted-state dataclasses live beside their owning transformers; only the four
scalers share a state model. They have no Pydantic dependency. Use
`dataclasses.asdict` and ordinary JSON serialization. State schema version `2`
includes `feature_names` and `categorical_features`, which are checked on replay.
Finite values, positive scales, vocabulary membership, ordered quantiles, support
counts and search budgets remain validated. Unknown/missing fields are rejected.
The former array API, `federated.preprocessing` imports, Pydantic recipe methods
and version-1 recipes are intentionally not supported.

The numerical Exareme3 steps accept one source and one caller-named derived
output. They retain source values and metadata. Ordinary execution keeps fitted
state internally; no public recipe-return transport is introduced. Python-fitted
state can be supplied for replay. The numerical imputer exposes mean/median/
constant only; categorical mode remains a Python capability requiring trusted
metadata vocabulary, never vocabulary inferred from observed values.

## Validation

Tests retain the original numerical assertions against NumPy, SciPy and
scikit-learn, plus partition stability, missing-value preservation, collective
categorical rejection, bounded search/refinement, malformed state and JSON replay.
Additional tests exercise the actual named DataFrame interface, existing
composition APIs and controller/worker derived-column execution. They do not
replace environment-backed deployment testing.
