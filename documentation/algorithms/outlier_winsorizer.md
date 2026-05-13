# Outlier Winsorizer

## Overview

`outlier_winsorizer` is an Exareme3 preprocessing step that clips selected
numerical variables before an algorithm runs. Bounds are computed on the local
worker data loaded for the request.

## Parameters

- `strategies`: required dictionary mapping variable names to `gaussian`, `iqr`,
  `mad`, or `quantile`.
- `tails`: optional dictionary mapping variable names to `left`, `right`, or
  `both`. The default is `both`.
- `folds`: optional dictionary mapping variable names to numeric fold values.
  Defaults are `gaussian=3.0`, `iqr=1.5`, `mad=3.0`, and `quantile=0.05`.

## Behavior

Configured variables must be numerical and present in `x` or `y`. Rows with
missing values in configured variables are dropped before bounds are computed.
The preprocessing step clips only configured variables and leaves all other
variables unchanged.

If the local worker data does not have enough non-missing values to compute
bounds for a configured variable, execution raises an insufficient data error.
