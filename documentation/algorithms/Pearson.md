## Pearson Correlation

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

#### Algorithm Description

This algorithm computes the Pearson correlation coefficient between two vectors *x* and
*y* using the eq.(1)

![pseudo](images/pearson_pseudocode.png)

#### Exareme3 Notes

- If `x_vars` is empty, the algorithm correlates `y` against itself.
- Returns correlation matrix, p-values, and confidence intervals (controlled by `alpha`).

<b><h4>Algorithm Implementation</b></h4>

[Pearson](../../exaflow/algorithms/exareme3/pearson_correlation.py)
