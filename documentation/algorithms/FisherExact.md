# Fisher's Exact Test

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

## Statistical model

For a 2 by 2 table:

```text
          y1   y2
x1        a    b
x2        c    d
```

the odds ratio is:

```text
OR = (a * d) / (b * c)
```

The p-value is computed from the hypergeometric distribution over tables with
the same margins.

## Federated computation

The test is computed without sharing row-level data. Each site contributes a
2 by 2 table aligned to the same factor and outcome categories. The final
statistic is computed from the aggregated table.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Outcome categories | Ensure exactly two aligned columns. |
| Factor categories | Ensure exactly two aligned rows. |
| 2 by 2 cell counts | Compute odds ratio and exact p-value. |

### Federated flow

```text
Input:
    y: binary outcome
    x: binary factor

Step 1:
    Determine the globally observed categories for y and x.

Step 2:
    Validate that the resulting table is 2 by 2.

Step 3:
    At each site:
        build a 2 by 2 contingency table with the global category order

Step 4:
    Aggregate the four cell counts.

Step 5:
    Compute odds ratio and exact p-value from the aggregated table.

Output:
    odds ratio, p-value, and table labels
```

## Technical decisions

- The exact test is applied to the aggregated 2 by 2 table.
- Category alignment is performed before aggregation.
- No continuity correction is applied by this implementation.
- The SciPy default two-sided alternative is used.

## Outputs

| Field | Description |
|---|---|
| `odds_ratio` | Estimated odds ratio. |
| `p_value` | Exact p-value. |
| `x_labels` | Factor-level row labels. |
| `y_labels` | Outcome-level column labels. |

## Validation against state-of-the-art implementation

Standalone tests compare the aggregated 2 by 2 table with:

```text
scipy.stats.fisher_exact(cross_tab)
```

Reference behavior is aligned with SciPy's default Fisher exact test.

## Limitations and assumptions

- Both variables must be binary.
- The test is intended for a 2 by 2 contingency table.
- The p-value conditions on fixed row and column margins.
- The odds ratio can be infinite or undefined when cells contain zeros.
