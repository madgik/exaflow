# Linear Regression

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

Linear regression fits an ordinary least squares model for a numerical outcome
using numerical and/or categorical covariates. The result includes coefficients,
standard errors, t-tests, p-values, confidence intervals, goodness-of-fit
statistics, and information criteria.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Single numerical outcome variable. |
| `x` | One or more numerical or categorical covariates. |

### Parameters

No user parameters are exposed for this algorithm.

## Statistical model

The model is ordinary least squares with an intercept:

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
computed from the aggregated cross-products.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Global categorical levels | Align one-hot encoded columns. |
| `X'X` | Estimate coefficients, covariance matrix, and rank. |
| `X'y` | Estimate coefficients. |
| Number of observations | Compute degrees of freedom and information criteria. |
| `sum(y)` and `sum(y^2)` | Compute total sum of squares. |
| Residual sum of squares | Compute residual standard error, R-squared, F statistic, and log-likelihood. |
| Sum of absolute residuals | Stored as a diagnostic quantity. |

### Federated flow

```text
Input:
    y: numerical outcome
    X: numerical and/or categorical covariates

Step 1:
    Align categorical levels across sites.

Step 2:
    Build the design matrix:
        one-hot encode categorical covariates
        include numerical covariates directly
        add an intercept

Step 3:
    At each site, compute:
        X'X
        X'y
        n
        sum(y)
        sum(y^2)

Step 4:
    Aggregate sufficient statistics.

Step 5:
    Estimate coefficients with a pseudo-inverse of X'X.

Step 6:
    At each site, compute residual contributions using the shared coefficients.

Step 7:
    Aggregate residual sum of squares and compute model summaries.

Output:
    coefficients, inferential statistics, and model fit statistics
```

## Technical decisions

- An intercept is always included.
- Categorical covariates are one-hot encoded with globally aligned categories.
- The pseudo-inverse of `X'X` is used, which allows rank-deficient designs to be
  handled numerically.
- Standard errors are computed from the diagonal of `(X'X)^+ * RSE^2`.
- Overall F-test display fields distinguish finite, undefined, and perfect-fit
  cases.
- Missing values are handled before fitting by the required missing-values
  preprocessing step.

## Outputs

| Field | Description |
|---|---|
| `dependent_var` | Outcome variable name. |
| `n_obs` | Number of observations used for fitting. |
| `df_resid` | Residual degrees of freedom. |
| `df_model` | Model degrees of freedom. |
| `rse` | Residual standard error. |
| `r_squared` | R-squared. |
| `r_squared_adjusted` | Adjusted R-squared. |
| `f_stat` | Overall model F statistic. |
| `f_pvalue` | P-value for the overall F statistic. |
| `ll` | Gaussian log-likelihood. |
| `aic` | Akaike information criterion. |
| `bic` | Bayesian information criterion. |
| `indep_vars` | Feature names including the intercept. |
| `coefficients` | Estimated coefficients. |
| `std_err` | Coefficient standard errors. |
| `t_stats` | Coefficient t statistics. |
| `pvalues` | Coefficient p-values. |
| `lower_ci`, `upper_ci` | Confidence interval bounds. |

## Validation against state-of-the-art implementation

Standalone tests compare results with `statsmodels` OLS:

```text
X_sm = sm.add_constant(X, has_constant="add")
sm.OLS(y, X_sm).fit()
```

Reference behavior is aligned with OLS using an intercept, while the computation
uses aggregated cross-products rather than centralized row-level data.

## Limitations and assumptions

- The outcome must be numerical.
- Observations are assumed independent.
- Standard inference assumes homoscedastic, approximately normal residuals.
- Strong multicollinearity or rank deficiency can affect interpretability even
  though the pseudo-inverse returns coefficients.
- No regularization is applied.
