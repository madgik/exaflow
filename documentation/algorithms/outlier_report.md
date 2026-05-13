# Outlier Report

## Overview

`outlier_report` is a diagnostic Exareme3 algorithm that reports local
per-dataset outliers for selected numerical variables. It does not modify the
input data. Use it before applying the `outlier_winsorizer` preprocessing step.

The algorithm computes winsorization bounds locally on each worker dataset and
reports how many values fall outside those bounds.

## Inputs

- `y`: one or more variables to inspect.
- `x`: optional additional variables to inspect.

Configured variables must be numerical. Categorical variables are rejected with
a user-facing validation error.

## Parameters

- `strategies`: required dictionary mapping variable names to one of:
  - `gaussian`: mean plus/minus `fold * std`
  - `iqr`: Q1/Q3 plus/minus `fold * IQR`
  - `mad`: median plus/minus `fold * normalized MAD`
  - `quantile`: lower and upper quantile caps
- `tails`: optional dictionary mapping variable names to `left`, `right`, or
  `both`. The default is `both`.
- `folds`: optional dictionary mapping variable names to numeric fold values.
  Defaults are `gaussian=3.0`, `iqr=1.5`, `mad=3.0`, and `quantile=0.05`.

## Output

The response contains a `featurewise` list with one record per configured
variable per local dataset. Each record includes the selected strategy, tail,
fold, computed bounds, and privacy-aware outlier counts.

The report intentionally omits descriptive statistics such as mean, standard
deviation, min, max, quartiles, and missing counts. Use `describe` for those.

## Privacy

If a variable/dataset does not have enough non-missing values to compute bounds,
the algorithm raises an insufficient data error.

Small non-zero outlier counts are suppressed as `null` according to the worker
minimum row count. A returned `0` means no outliers were detected.

## Related preprocessing

`outlier_winsorizer` uses the same `strategies`, `tails`, and `folds`
configuration to clip values locally per dataset before the selected algorithm
runs.
