## Descriptive Statistics

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

#### Overview

Computes descriptive statistics for numerical and categorical variables in a
federated setting. Numerical variables report counts, mean, standard deviation,
quantiles, and min/max. Categorical variables report counts per category.

#### Exareme3 Notes

- Uses federated descriptive statistics aggregation.
- Does not drop NA rows and does not enforce minimum-row checks (see algorithm code).
- Adds a dataset identifier column to support per-dataset summaries.

#### Algorithm Implementation

[Describe](../../exaflow/algorithms/exareme3/describe.py)
