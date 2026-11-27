# Standardized Mean Difference

Runtime wrapper: [statistics/standardized_mean_difference.py](../../exaflow/algorithms/exareme3/statistics/standardized_mean_difference.py).

Its federated core remains in the `statistics` family.

## Overview

Standardized Mean Difference computes Cohen's d for every unique pair of categories in one grouping variable. It reports the size and direction of each difference relative to the groups' pooled within-group variability.

## Inputs

- **Grouping variable (`x`)**: one nominal text or integer variable. Its observed categories define the comparisons.
- **Outcome (`y`)**: one numerical variable whose means are compared.

Rows with a missing group label or outcome value do not contribute. A category must reach the configured privacy minimum to appear in a comparison.

## Method

The mathematical method, library API, aggregation flow, numerical decisions, reference comparisons, and limitations are documented in [Standardized mean difference](../../exaflow/algorithms/federated/docs/standardized_mean_difference.md).

## Application integration

The wrapper supplies the configured privacy threshold for group eligibility rather than the library default of two observations. Fewer than two eligible groups produce the application insufficient-data error.

## Outputs

A successful response contains `comparisons`, a list whose entries contain:

- `group1`, `group2`: compared category labels;
- `smd`: pooled-variance Cohen's d.

When fewer than two eligible groups remain, the analysis fails with an insufficient-data error. `POST /analysis` returns HTTP 461, and RabbitMQ reports `insufficient_data`. Previously, this case returned a successful response with an empty `comparisons` list.
