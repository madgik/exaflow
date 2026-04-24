# Welch's T-Test

## Overview

The `ttest_welch` algorithm performs a federated Welch's independent t-test for
comparing the means of two independent groups when equal variances should not be
assumed.

The output contains:

- `t_stat`: Welch t statistic for `groupA - groupB`.
- `df`: Welch-Satterthwaite degrees of freedom.
- `p`: p-value for the selected alternative hypothesis.
- `mean_diff`: difference in group means.
- `se_diff`: standard error of the mean difference.
- `ci_lower` and `ci_upper`: confidence interval bounds for the mean difference.
- `cohens_d`: standardized mean difference using the average group variance.

## API Contract

- **Algorithm name**: `ttest_welch`
- **Type**: `exareme3`
- **Status**: enabled

### Inputs

- `inputdata.y`: exactly one numerical variable.
- `inputdata.x`: exactly one nominal grouping variable.

### Parameters

- `groupA`: grouping-variable category used as the first sample.
- `groupB`: grouping-variable category used as the second sample.
- `alt_hypothesis`: `two-sided`, `less`, or `greater`.
- `alpha`: significance level used for confidence interval calculation.

## Validation Notes

- Standalone parity tests compare the federated implementation with a
  centralized Welch reference.
- Production validation verifies the algorithm executes and returns the expected
  result schema for a dementia data-model fixture.
