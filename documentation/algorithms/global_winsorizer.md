# Global Winsorizer

## Overview

Create one derived column before the downstream analysis while retaining the source.
The specification is available through `GET /specifications/preprocessing` when its
required components are enabled. It adds no new API route.

## Inputs

Select one `source_variable` available at this stage of the request, and provide
exactly one unused name in `new_column_names`. The source must be numerical. There is no generated default name.

### Parameters

`lower_quantile` defaults to `0.01` and `upper_quantile` to `0.99`, with `0 <= lower < upper <= 1`. Optional `recipe` replays fitted bounds.

Example element of the request's `preprocessing` list:

```json
{
  "name": "global_winsorizer",
  "parameters": {
    "source_variable": "crp",
    "new_column_names": [
      "crp__prepared"
    ]
  }
}
```

## Statistical method

Clip derived values to the two approximate pooled quantile bounds.

## Computation without row-level data

Only local values are transformed. Fitted numerical parameters use aggregated
quantities rather than transferring raw observations.

### Aggregated quantities

Existing aggregated percentile estimates, shared across all participants.

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

A real-valued clipped column is appended. The source and all rows are preserved; NaNs remain NaN. Neither fitting nor transformation writes to the stored datasets.

## Validation against a reference method

Actual clipped values are compared against the fitted global bounds, including an extreme observation and uneven partitions.

## Limitations and assumptions

Percentiles are approximate. This differs from the existing local outlier_winsorizer, whose bounds are estimated separately on each participant. Participants must use the same ordered source and method settings.
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
