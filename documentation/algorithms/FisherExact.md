# Fisher's Exact Test

## Overview

Fisher's exact test evaluates association between two binary categorical
variables using the exact probability of observing a 2 by 2 contingency table
under fixed margins.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Binary nominal outcome variable. |
| `x` | Binary nominal factor variable. |

### Parameters

No user parameters are exposed for this algorithm.

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Fisher's exact test](../../exaflow/algorithms/federated/docs/fisher_exact.md).

## Outputs

| Field | Description |
|---|---|
| `odds_ratio` | Estimated odds ratio. |
| `p_value` | Exact p-value. |
| `x_labels` | Factor-level row labels. |
| `y_labels` | Outcome-level column labels. |
