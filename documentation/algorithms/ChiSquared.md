## Chi-Squared Test of Independence

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm's global step.

#### Overview

Tests whether two categorical variables are statistically independent by comparing
observed cell counts against expected counts under the null hypothesis. Supports
contingency tables of any size (R×C).

#### Algorithm Description

Each worker builds its local cell counts for the (factor, outcome) pair and sends
them to the aggregation server. The controller assembles the global contingency
table and runs `scipy.stats.chi2_contingency`.

The **test statistic** and **degrees of freedom** are:

`χ² = Σ_ij [ (O_ij − E_ij)² / E_ij ]` where `E_ij = (row_i × col_j) / n`

`dof = (R − 1) × (C − 1)`

#### Inputs

| Field | Type | Required | Description |
|-------|--------|----------|-------------|
| `x` | `TEXT` | Yes | Factor (independent variable) — nominal, 2 or more categories |
| `y` | `TEXT` | Yes | Outcome (dependent variable) — nominal, 2 or more categories |

#### Outputs

| Field | Type | Description |
|------------|---------------------|-------------|
| `chi2` | `float` | Chi-Squared test statistic. |
| `p_value` | `float` | Asymptotic p-value. |
| `dof` | `int` | Degrees of freedom: `(R−1) × (C−1)`. |
| `expected` | `list[list[float]]` | Expected frequency matrix (R×C) under independence. |
| `x_labels` | `list[str]` | Factor category labels (row order). |
| `y_labels` | `list[str]` | Outcome category labels (column order). |

#### Exareme3 Notes

- Rows with missing values are always dropped; `NaN` cannot be treated as a category.
- For small 2×2 tables with low expected counts, prefer Fisher's Exact Test.

<b><h4>Algorithm Implementation</b></h4>

[Chi-Squared Test](../../exaflow/algorithms/exareme3/chi_squared.py)
