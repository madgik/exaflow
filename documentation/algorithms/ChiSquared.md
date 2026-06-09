# Chi-squared Test

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Statistical model](#statistical-model)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

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

## Statistical model

For a contingency table with observed counts `O_ij`, expected counts under
independence are:

```text
E_ij = row_sum_i * column_sum_j / n
```

The test statistic is:

```text
chi2 = sum_ij (O_ij - E_ij)^2 / E_ij
```

Degrees of freedom are:

```text
(rows - 1) * (columns - 1)
```

## Federated computation

The test is computed without sharing row-level data. Each site builds a
contingency table using globally aligned category labels. Cell counts are
summed, and the chi-squared statistic is computed from the aggregated table.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Outcome categories | Align contingency-table columns. |
| Factor categories | Align contingency-table rows. |
| Cell counts | Build the aggregated contingency table. |
| Row and column totals | Compute expected counts under independence. |

### Federated flow

```text
Input:
    y: categorical outcome
    x: categorical factor

Step 1:
    Determine the globally observed categories for y and x.

Step 2:
    At each site:
        build a contingency table with the global row and column order
        include zero-count cells where a category is absent

Step 3:
    Aggregate contingency-table cell counts.

Step 4:
    Compute expected counts under independence.

Step 5:
    Compute chi-squared statistic, degrees of freedom, and p-value.

Output:
    observed table, expected table, statistic, p-value, and degrees of freedom
```

## Technical decisions

- Category alignment is performed before count aggregation.
- Missing values may be represented as table categories when included by the
  cross-tabulation helper.
- The test uses the aggregated contingency table, not per-site test statistics.
- Expected counts are derived after aggregation.

## Outputs

| Field | Description |
|---|---|
| `chi2` | Chi-squared statistic. |
| `p` | P-value. |
| `dof` | Degrees of freedom. |
| `expected` | Expected counts under independence. |
| `x_labels` | Factor-level row labels. |
| `y_labels` | Outcome-level column labels. |

## Validation against state-of-the-art implementation

Standalone tests compare the aggregated contingency table result with:

```text
scipy.stats.chi2_contingency(cross_tab)
```

Reference behavior is aligned with SciPy's chi-squared test on the centralized
contingency table.

## Limitations and assumptions

- Both variables must be categorical.
- The usual chi-squared approximation assumes sufficiently large expected cell
  counts.
- Sparse tables can make the asymptotic p-value unreliable.
- The test detects association but does not estimate effect direction or causal
  relationships.
