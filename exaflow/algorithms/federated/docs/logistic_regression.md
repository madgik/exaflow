# Binary logistic regression

Implementation: [linear_model/logistic_regression.py](../linear_model/logistic_regression.py).

Binary logistic regression estimates a probability from a numeric design matrix and an already binary-encoded outcome.

## Library interface

`FederatedLogisticRegression(fit_intercept=True).fit(X, y, *, agg_client)` requires numeric features and outcomes already encoded as `0` or `1`. The intercept flag is keyword-only. The result exposes `params`, `stderr`, `z_scores`, `pvalues`, `lower_ci`, `upper_ci`, `ll`, `ll0`, `r_squared_cs`, `r_squared_mcf`, `aic`, `bic`, degrees of freedom, and `predict(X)` probabilities. Iteration limit 50, tolerance `1e-4`, and significance level `0.05` are module constants, not constructor options. Encoding belongs to separate [one-hot](one_hot_encoder.md) and [column transformation](column_transformer.md) components.

## Statistical method

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
| Number of observations | Validate model size and compute fit statistics. |
| Number of positive outcomes | Validate class support and compute null log-likelihood. |
| Gradient vector | Update coefficients. |
| Hessian matrix | Compute Newton step and coefficient covariance approximation. |
| Log-likelihood | Compute convergence diagnostics and model fit statistics. |

### Federated flow

```text
Receive numeric X and binary y; add an intercept when configured.
Validate effective inputs, then aggregate total and per-class sample counts.
Initialize beta to zero.
Repeat up to 50 iterations:
    compute local probabilities, score, Hessian, and log-likelihood
    sum these quantities across participants
    stop if the largest absolute score component is <= 1e-4
    update beta with a Newton step (pseudo-inverse fallback)
Compute Wald uncertainty, likelihood statistics, and fitted probabilities.
```

## Numerical decisions

- The optimizer uses Newton updates.
- `MAX_ITER` is 50 and convergence tolerance is `1e-4` on the maximum absolute
  gradient component.
- If the Hessian is singular, a pseudo-inverse is used.
- Validation requires enough observations and enough positive/negative outcomes
  relative to the number of predictors.
- Wald standard errors and confidence intervals use the inverse Hessian.
- An intercept is included only when `fit_intercept=True`. Features and encoded targets must be finite and row-aligned. Global observation and class counts must exceed the fitted parameter count.

## Reference comparison

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

## Limitations

- Observations are assumed independent.
- No regularization is applied.
- P-values and confidence intervals rely on large-sample Wald approximations.
- Complete or quasi-complete separation can prevent convergence.
- The model can fail when observations or class counts are too small relative to
  the number of predictors.
