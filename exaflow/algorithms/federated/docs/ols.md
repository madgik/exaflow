# Ordinary least squares

Implementation: [linear_model/ols.py](../linear_model/ols.py).

Linear regression fits an ordinary least squares model for a numerical outcome
using a numerical design matrix, with categorical covariates encoded separately. The result includes coefficients,
standard errors, t-tests, p-values, confidence intervals, goodness-of-fit
statistics, and information criteria.

## Library interface

`FederatedOLS(fit_intercept=True).fit(X, y, *, agg_client)` accepts a numeric design matrix and outcome. The intercept flag is keyword-only. The fitted result exposes `params`, `bse`, `tvalues`, `pvalues`, `df_model`, `df_resid`, `rse`, `fvalue`, `f_pvalue`, `rsquared`, `rsquared_adj`, `ll`, `aic`, `bic`, `nobs`, residual sums, covariance information, and `predict(X)`. Categorical encoding is a separate component; see [column transformation](column_transformer.md) and [one-hot encoding](one_hot_encoder.md).

## Statistical method

The model is ordinary least squares; with an intercept it is:

```text
y = beta_0 + beta_1 x_1 + ... + beta_p x_p + epsilon
```

Coefficients minimize the residual sum of squares:

```text
RSS(beta) = sum_i (y_i - x_i beta)^2
```

The fitted coefficients are computed from the normal equations using a
pseudo-inverse:

```text
beta = (X'X)^+ X'y
```

## Federated computation

The model is computed without sharing row-level data. Each site contributes OLS
sufficient statistics, and the coefficient vector and summary statistics are
computed from the aggregated cross-products. The total sum of squares uses a
second pass around the global outcome mean instead of raw moments
(`sum(y^2) - n * mean^2`), which lose all precision when the mean is large
relative to the spread.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| `X'X` | Estimate coefficients, covariance matrix, and rank. |
| `X'y` | Estimate coefficients. |
| Number of observations | Compute degrees of freedom and information criteria. |
| `sum(y)` | Compute the global outcome mean. |
| `sum((y - mean_y)^2)` | Compute total sum of squares. |
| Residual sum of squares | Compute residual standard error, R-squared, F statistic, and log-likelihood. |
| Sum of absolute residuals | Stored as a diagnostic quantity. |

### Federated flow

```text
Receive an aligned numeric design matrix and outcome.
Add an intercept when configured.
Sum X'X, X'y, counts, and outcome moments across participants.
Solve beta = pinv(X'X) X'y.
Compute residual statistics and coefficient covariance from aggregated quantities.
Return the fitted results object; predict locally using X beta.
```

## Numerical decisions

- The pseudo-inverse of `X'X` is used, which allows rank-deficient designs to be
  handled numerically.
- Standard errors are computed from the diagonal of `(X'X)^+ * RSE^2`.
- Intercept inclusion follows `fit_intercept`. Categorical encoding is supplied separately. Undefined and perfect-fit F diagnostics retain their numerical representation in the result.

## Reference comparison

Standalone tests compare results with `statsmodels` OLS:

```text
X_sm = sm.add_constant(X, has_constant="add")
sm.OLS(y, X_sm).fit()
```

Reference behavior is aligned with OLS using an intercept, while the computation
uses aggregated cross-products rather than centralized row-level data.

## Limitations

- The outcome must be numerical.
- Observations are assumed independent.
- Standard inference assumes homoscedastic, approximately normal residuals.
- Strong multicollinearity or rank deficiency can affect interpretability even
  though the pseudo-inverse returns coefficients.
- No regularization is applied.
