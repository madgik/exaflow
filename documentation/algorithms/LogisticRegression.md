# Logistic Regression

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

Logistic regression models a binary outcome as a function of numerical and/or
categorical covariates. The selected positive class is encoded as `1`; all other
outcome classes are encoded as `0`.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Nominal outcome variable converted to binary using `positive_class`. |
| `x` | One or more numerical or categorical covariates. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `positive_class` | Outcome category treated as the positive outcome. | Required |

## Statistical model

The model is binary logistic regression with a logit link:

```text
P(y_i = 1 | x_i) = sigmoid(eta_i)
eta_i = beta_0 + beta_1 x_i1 + ... + beta_p x_ip
sigmoid(eta) = 1 / (1 + exp(-eta))
```

Coefficients maximize the binomial log-likelihood:

```text
LL(beta) = sum_i [y_i log(mu_i) + (1 - y_i) log(1 - mu_i)]
```

## Federated computation

The model is computed without sharing row-level data. Each site contributes
gradient, Hessian, and log-likelihood terms for the current coefficient vector.
Aggregated quantities drive Newton-style coefficient updates.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Global categorical levels | Align one-hot encoded covariate columns. |
| Number of observations | Validate model size and compute fit statistics. |
| Number of positive outcomes | Validate class support and compute null log-likelihood. |
| Gradient vector | Update coefficients. |
| Hessian matrix | Compute Newton step and coefficient covariance approximation. |
| Log-likelihood | Compute convergence diagnostics and model fit statistics. |

### Federated flow

```text
Input:
    y: nominal outcome
    X: covariates
    positive_class: category mapped to 1

Step 1:
    Convert y to binary:
        1 if y equals positive_class
        0 otherwise

Step 2:
    Build the design matrix:
        one-hot encode categorical covariates
        include numerical covariates directly
        add an intercept

Step 3:
    Aggregate validation quantities:
        total observations
        total positive outcomes

Step 4:
    Initialize coefficients to zero.

Step 5:
    Repeat up to the maximum iteration count:
        each site computes fitted probabilities
        each site computes local gradient, Hessian, and log-likelihood
        aggregate gradient, Hessian, and log-likelihood
        invert the Hessian, using a pseudo-inverse if needed
        update coefficients with the Newton step
        stop when max absolute gradient <= tolerance

Step 6:
    Compute standard errors, z statistics, p-values, confidence intervals,
    pseudo R-squared values, AIC, and BIC.

Output:
    coefficients and model summary
```

## Technical decisions

- An intercept is always included.
- Categorical covariates are one-hot encoded with globally aligned categories.
- The optimizer uses Newton updates.
- `MAX_ITER` is 50 and convergence tolerance is `1e-4` on the maximum absolute
  gradient component.
- If the Hessian is singular, a pseudo-inverse is used.
- Validation requires enough observations and enough positive/negative outcomes
  relative to the number of predictors.
- Wald standard errors and confidence intervals use the inverse Hessian.
- Missing values are handled before fitting by the required missing-values
  preprocessing step.

## Outputs

| Field | Description |
|---|---|
| `n_obs` | Number of observations used for fitting. |
| `coefficients` | Estimated coefficients including intercept. |
| `stderr` | Standard errors. |
| `lower_ci`, `upper_ci` | Confidence interval bounds. |
| `z_scores` | Wald z statistics. |
| `pvalues` | Two-sided coefficient p-values. |
| `df_model` | Model degrees of freedom. |
| `df_resid` | Residual degrees of freedom. |
| `r_squared_cs` | Cox-Snell pseudo R-squared. |
| `r_squared_mcf` | McFadden pseudo R-squared. |
| `ll0` | Null-model log-likelihood. |
| `ll` | Fitted-model log-likelihood. |
| `aic` | Akaike information criterion. |
| `bic` | Bayesian information criterion. |

## Validation against state-of-the-art implementation

Standalone tests compare with `statsmodels`:

```text
smf.logit(formula, df).fit(
    method="newton",
    maxiter=200,
    tol=1e-8,
    start_params=start_params,
    disp=False,
)
```

Reference behavior is aligned with `statsmodels` logistic regression using a
Newton optimizer and an intercept.

## Limitations and assumptions

- The dependent variable is binarized by the selected positive class.
- Observations are assumed independent.
- No regularization is applied.
- P-values and confidence intervals rely on large-sample Wald approximations.
- Complete or quasi-complete separation can prevent convergence.
- The model can fail when observations or class counts are too small relative to
  the number of predictors.
