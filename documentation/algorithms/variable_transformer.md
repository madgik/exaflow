# Variable Transformer

## Overview

Create one derived column before the downstream analysis while retaining the source.
The specification is available through `GET /specifications/preprocessing` when its
required components are enabled. It adds no new API route.

## Inputs

Select one `source_variable` available at this stage of the request, and provide
exactly one unused name in `new_column_names`. The source must be numerical. There is no generated default name.

### Parameters

`method`: log1p, sqrt, asinh, yeo_johnson, quantile_uniform or quantile_normal. Yeo-Johnson defaults: 64 evaluations, lambda tolerance `1e-5`, likelihood tolerance `1e-10`. Quantile defaults: 100 quantiles, 256 bins, two extra refinement rounds and configured interval support five. The platform floor cannot be lowered. Optional `recipe` supports fitted power/quantile replay.

Example element of the request's `preprocessing` list:

```json
{
  "name": "variable_transformer",
  "parameters": {
    "source_variable": "crp",
    "new_column_names": [
      "crp__prepared"
    ],
    "method": "log1p"
  }
}
```

## Statistical method

Elementwise log1p/sqrt require nonnegative values; asinh accepts all reals. Yeo-Johnson fits the Jacobian-adjusted profile likelihood. Quantile methods interpolate approximate global ranks and optionally apply the inverse normal CDF.

## Computation without row-level data

Only local values are transformed. Fitted numerical parameters use aggregated
quantities rather than transferring raw observations.

### Aggregated quantities

Domain-error counts; power-fit extrema, count/Jacobian and corrected variance terms; or packed quantile histograms and supported-group extrema.

### Federated flow

```text
validate the source and the declared output name
fit common parameters from aggregated quantities, or replay validated state
transform the selected source locally
append the derived column and its declared numerical metadata
make both the source and derived column available downstream
```

## Technical decisions

The implementation is a direct `PreprocessingStep` subclass and uses the existing
base aggregation injection, registry discovery and `outputs` validation. The
mathematics stays in domain-organized federated transformers implementing the
named-column `FederatedTransformer` interface. Replay state uses validated
version-2 dataclasses internally; ordinary request/specification DTOs remain
Pydantic. Recipes are not returned publicly by preprocessing; fit-only settings
apply to fitting, while a supplied recipe is authoritative for replay.

## Outputs

A real-valued derived column is appended. The source, its metadata and missing values remain unchanged. Neither fitting nor transformation writes to the stored datasets.

## Validation against a reference method

Elementwise NumPy reference results, SciPy Yeo-Johnson comparison, skewed-distribution regressions, partition-stable fits and exact replay tests.

## Limitations and assumptions

Power search is bounded and can report an approximate optimum. Quantile precision depends on achieved intervals, not a universal relative-error guarantee. Histogram information is not differential privacy. Participants must use the same ordered source and method settings.
Standalone tests exercise controller/worker code paths. The HTTP integration suite
`tests/prod_env_tests/test_derived_transformations.py` checks deployed steps with
real aggregation-backed workers and non-longitudinal fixtures.
See the [federated mathematics and fitted-state contract](../../exaflow/algorithms/federated/docs/transformations.md)
for numerical details, privacy limitations and the breaking Python/state changes.

## Compatibility and rollback

This is a source-preserving, single-output contract, not the unmerged catalog's
in-place `strategies` mapping. No legacy aliases are accepted. A downstream
algorithm must explicitly select the derived name to use it. Revert the related
wrappers, numerical interface/state changes and their tests together; no stored
data migration or deployment change is required.
