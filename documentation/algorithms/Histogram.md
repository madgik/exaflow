## Histogram

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

#### Overview

Computes federated histograms for a target variable, optionally grouped by
categorical variables. The bins are shared across workers and counts are
aggregated with privacy constraints.

#### Exareme3 Notes

- `y_var` is the variable to histogram; `x_vars` are optional grouping variables.
- `bins` defaults to 20 if not provided.
- Enforces a minimum row count per group via worker privacy config.

#### Algorithm Implementation

[Histogram](../../exaflow/algorithms/exareme3/histogram.py)
