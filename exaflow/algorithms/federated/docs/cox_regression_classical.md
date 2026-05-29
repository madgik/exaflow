# Federated Cox Regression Classical

## Overview

`FederatedClassicalCoxRegression` implements the classical Cox proportional
hazards model through the Cox partial likelihood with Breslow handling of ties.

## Federated Sufficient Statistics

For each global event time `t_j`, every worker computes:

- `d_j`: local number of events at `t_j`
- `E1_j = sum_{i in D_j} x_i`
- `S0_j(beta) = sum_{i in R_j} exp(x_i^T beta)`
- `S1_j(beta) = sum_{i in R_j} x_i exp(x_i^T beta)`
- `S2_j(beta) = sum_{i in R_j} x_i x_i^T exp(x_i^T beta)`

The controller aggregates these blocks across workers and forms:

- partial log-likelihood
- score vector
- observed information matrix

## Optimization

The current implementation keeps the Newton-Raphson logic inside
`cox_regression_classical.py`:

- regularized linear solve for the Newton direction
- step clipping through `max_step_norm`
- convergence on both coefficient movement and score norm

## Numerical Details

- event times are aggregated globally through exact union, with no binning
- risk-set sums are computed with reverse cumulative sums on each worker
- exponential weights are clipped before exponentiation for stability

## Benchmarking

The standalone test suite compares the federated result against:

- `statsmodels.PHReg(..., ties="breslow")`
- the existing stacked approximation `FederatedStackedCoxRegression`
