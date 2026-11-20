# Variable Scaler

## Overview

Create one derived column before the downstream analysis while retaining the source.
The specification is available through `GET /specifications/preprocessing` when its
required components are enabled. It adds no new API route.

## Inputs

Select one `source_variable` available at this stage of the request, and provide
exactly one unused name in `new_column_names`. The source must be numerical. There is no generated default name.

### Parameters

`method`: standard, robust, minmax or maxabs. `feature_range` is a two-number min-max range, default `[0, 1]`. Optional `recipe` replays a matching one-feature fitted state.

Example element of the request's `preprocessing` list:

```json
{
  "name": "variable_scaler",
  "parameters": {
    "source_variable": "crp",
    "new_column_names": [
      "crp__prepared"
    ],
    "method": "minmax"
  }
}
```

## Statistical method

Standard uses `(x-mean)/population_scale`; robust uses `(x-median)/IQR`; min-max maps pooled extrema to the target range; max-abs divides by the pooled maximum magnitude. Degenerate scales use one.

## Computation without row-level data

Only local values are transformed. Fitted numerical parameters use aggregated
quantities rather than transferring raw observations.

### Aggregated quantities

Sums/counts and corrected centred sums for standard scaling; shared percentiles for robust scaling; extrema for min-max and max-abs.

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

The original source is retained. The new output is real-valued; source bounds/units are not copied to it. Missing values remain missing. Neither fitting nor transformation writes to the stored datasets.

## Validation against a reference method

NumPy/scikit-learn standard, min-max and max-abs reference results, plus stated-tolerance pooled quartile comparisons for robust scaling.

## Limitations and assumptions

Robust quartiles are approximate. Min-max replay can produce values outside the fitted range for later observations. Participants must use the same ordered source and method settings.
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
