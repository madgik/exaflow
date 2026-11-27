# Chi-squared Test

## Overview

The chi-squared test evaluates whether two categorical variables are independent.
It builds a contingency table for an outcome variable and a factor variable,
then compares observed cell counts with expected counts under independence.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Nominal outcome variable. |
| `x` | Nominal factor variable. |

### Parameters

No user parameters are exposed for this algorithm.

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Chi-squared test](../../exaflow/algorithms/federated/docs/chi_squared.md).

## Outputs

| Field | Description |
|---|---|
| `chi2` | Chi-squared statistic. |
| `p` | P-value. |
| `dof` | Degrees of freedom. |
| `expected` | Expected counts under independence. |
| `x_labels` | Factor-level row labels. |
| `y_labels` | Outcome-level column labels. |
