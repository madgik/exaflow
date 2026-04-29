## Fisher's Exact Test

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm's global step.

#### Overview

Tests whether there is a statistically significant association between two binary
categorical variables. Computes an **exact p-value** using the hypergeometric
distribution — no large-sample approximation needed, making it reliable for small
datasets.

#### Algorithm Description

Each worker builds its local 2×2 cell counts for the (factor, outcome) pair and
sends them to the aggregation server. The controller assembles the global
contingency table and runs `scipy.stats.fisher_exact`.

Given the 2×2 table with cells *a*, *b*, *c*, *d*, the **odds ratio** is:

`OR = (a × d) / (b × c)`

The **p-value** is the exact probability of observing a table at least as extreme
as the one seen, under the null hypothesis of independence.

#### Inputs

| Field | Type | Required | Description |
|-------|--------|----------|-------------|
| `x` | `TEXT` | Yes | Factor (independent variable) — nominal, exactly 2 categories |
| `y` | `TEXT` | Yes | Outcome (dependent variable) — nominal, exactly 2 categories |

#### Outputs

| Field | Type | Description |
|--------------|-------------|-------------|
| `odds_ratio` | `float` | Odds ratio of the 2×2 table. |
| `p_value` | `float` | Exact p-value of the test. |
| `x_labels` | `list[str]` | Factor category labels (row order). |
| `y_labels` | `list[str]` | Outcome category labels (column order). |

#### Exareme3 Notes

- Both variables must have **exactly 2 categories**. A `BadInputError` is raised if the resulting table is not 2×2.
- Rows with missing values are always dropped; `NaN` cannot be treated as a category.
- A zero cell count produces an undefined odds ratio (`inf` or `0`).
- The implementation uses SciPy's default **two-tailed** alternative.

<b><h4>Algorithm Implementation</b></h4>

[Fisher's Exact Test](../../exaflow/algorithms/exareme3/fisher_exact.py)
