# Outlier Detector

## Overview

Create one derived column before the downstream analysis while retaining the source.
The specification is available through `GET /specifications/preprocessing` when its
required components are enabled. It adds no new API route.

## Inputs

Select one `source_variable` available at this stage of the request, and provide
exactly one unused name in `new_column_names`. The source must be numerical. There is no generated default name.

### Parameters

`mode`: zscore or robust. Positive `threshold` defaults to three. Optional `recipe` replays matching fitted location and scale.

Example element of the request's `preprocessing` list:

```json
{
  "name": "outlier_detector",
  "parameters": {
    "source_variable": "crp",
    "new_column_names": [
      "crp__prepared"
    ],
    "mode": "zscore"
  }
}
```

## Statistical method

Flag `abs(x-mean)/population_scale > threshold`, or `abs(x-median) > threshold*IQR` in robust mode. Degenerate scales use one.

## Computation without row-level data

Only local values are transformed. Fitted numerical parameters use aggregated
quantities rather than transferring raw observations.

### Aggregated quantities

Corrected two-pass moment reductions or pooled approximate quartiles.

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

An integer zero/one flag is appended. Missing values get zero. The source and row set are unchanged. Neither fitting nor transformation writes to the stored datasets.

## Validation against a reference method

Fitted threshold behavior, degenerate scales, global fitting, integer output typing and replay regressions.

## Limitations and assumptions

Robust quartiles are approximate. The step is diagnostic only: drop/set-missing actions from the unmerged catalog are not included. Participants must use the same ordered source and method settings.
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
