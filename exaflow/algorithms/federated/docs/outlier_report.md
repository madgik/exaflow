# Outlier Report

## Family

- `statistics`

## Purpose

`FederatedOutlierReport` computes local per-dataset winsorization bounds and
outlier counts for configured numerical variables. It is diagnostic only and
does not alter input values.

## Strategies

- `gaussian`: mean plus/minus `fold * std`
- `iqr`: Q1/Q3 plus/minus `fold * IQR`
- `mad`: median plus/minus `fold * normalized MAD`
- `quantile`: lower and upper quantile caps

The helper also supports tail selection with `left`, `right`, and `both`.

## Privacy

The implementation raises `InsufficientDataError` when a local dataset has too
few non-missing values for a configured variable. Small non-zero outlier counts
are returned as `null`.
