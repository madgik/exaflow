from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults

VAR_SMOOTHING = 1e-9


class FederatedGaussianNBResults(FederatedEstimatorResults):
    """Results container for federated Gaussian Naive Bayes."""

    nobs: int

    def __init__(
        self,
        *,
        x_vars: List[str],
        labels: List[str],
        theta: np.ndarray,
        var: np.ndarray,
        class_count: np.ndarray,
        class_prior: np.ndarray,
        n_obs: int,
        var_smoothing: float,
    ) -> None:
        self.x_vars = list(x_vars)
        self.labels = list(labels)
        self.theta = np.asarray(theta, dtype=float)
        self.var = np.asarray(var, dtype=float)
        self.class_count = np.asarray(class_count, dtype=float)
        self.class_prior = np.asarray(class_prior, dtype=float)
        self.nobs = int(n_obs)
        self.var_smoothing = float(var_smoothing)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from scipy import stats as scipy_stats

        if self.theta is None or self.var is None or self.class_prior is None:
            raise ValueError("GaussianNB is not fitted yet.")

        X_enc = _ensure_numeric_features(X, self.x_vars)
        if X_enc.shape[0] == 0 or len(self.labels) == 0:
            return np.zeros((0, len(self.labels)), dtype=float)

        theta = self.theta[np.newaxis, :, :]
        sigma = np.sqrt(self.var)[np.newaxis, :, :]

        factors = scipy_stats.norm.pdf(X_enc[:, np.newaxis, :], loc=theta, scale=sigma)
        likelihood = factors.prod(axis=2)

        prior = self.class_prior[np.newaxis, :]
        unnormalized_post = prior * likelihood
        denom = unnormalized_post.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        posterior = unnormalized_post / denom
        return posterior

    def predict(self, X: np.ndarray) -> np.ndarray:
        posterior = self.predict_proba(X)
        if posterior.shape[0] == 0:
            return np.asarray([], dtype=object)
        idx = posterior.argmax(axis=1)
        labels_arr = np.asarray(self.labels, dtype=object)
        return labels_arr[idx]


class FederatedGaussianNB(FederatedEstimator):
    """Federated Gaussian Naive Bayes with sklearn-like interface."""

    def __init__(
        self,
        *,
        x_vars: List[str],
        labels: List[str],
        var_smoothing: float = VAR_SMOOTHING,
    ) -> None:
        self.x_vars = list(x_vars)
        self.labels = list(labels)
        self.var_smoothing = float(var_smoothing)
        self.results: FederatedGaussianNBResults | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        agg_client: AggregationClient,
    ) -> FederatedGaussianNBResults:
        X_enc, y_arr = _ensure_training_data(X, y, self.x_vars)
        n_classes = len(self.labels)
        n_features = X_enc.shape[1] if X_enc.ndim == 2 else len(self.x_vars)

        counts_local = np.zeros((n_classes, n_features), dtype=float)
        sums_local = np.zeros_like(counts_local)
        sums_sq_local = np.zeros_like(counts_local)

        if y_arr.size:
            for class_idx, label in enumerate(self.labels):
                mask = y_arr == label
                if not np.any(mask):
                    continue
                X_class = X_enc[mask]
                counts_local[class_idx, :] = np.sum(~np.isnan(X_class), axis=0)
                sums_local[class_idx, :] = np.nansum(X_class, axis=0)
                sums_sq_local[class_idx, :] = np.nansum(X_class**2, axis=0)

        counts_global = np.asarray(agg_client.sum(counts_local), dtype=float)
        sums_global = np.asarray(agg_client.sum(sums_local), dtype=float)
        sums_sq_global = np.asarray(agg_client.sum(sums_sq_local), dtype=float)

        if counts_global.size == 0:
            results = FederatedGaussianNBResults(
                x_vars=self.x_vars,
                labels=[],
                theta=np.zeros((0, n_features), dtype=float),
                var=np.zeros((0, n_features), dtype=float),
                class_count=np.zeros(0, dtype=float),
                class_prior=np.zeros(0, dtype=float),
                n_obs=0,
                var_smoothing=self.var_smoothing,
            )
            self.results = results
            return results

        class_count_full = counts_global[:, 0]
        total_n_obs = float(class_count_full.sum())

        keep_mask = class_count_full > 0
        if not np.any(keep_mask):
            results = FederatedGaussianNBResults(
                x_vars=self.x_vars,
                labels=[],
                theta=np.zeros((0, n_features), dtype=float),
                var=np.zeros((0, n_features), dtype=float),
                class_count=np.zeros(0, dtype=float),
                class_prior=np.zeros(0, dtype=float),
                n_obs=0,
                var_smoothing=self.var_smoothing,
            )
            self.results = results
            return results

        counts_eff = counts_global[keep_mask, :]
        sums_eff = sums_global[keep_mask, :]
        sums_sq_eff = sums_sq_global[keep_mask, :]

        means = sums_eff / counts_eff
        var = (
            sums_sq_eff - 2 * means * sums_eff + counts_eff * (means**2)
        ) / counts_eff

        var_max = var.max() if var.size else 0.0
        epsilon = self.var_smoothing * var_max
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            epsilon = self.var_smoothing
        var = np.clip(var, epsilon, None)

        class_count_eff = class_count_full[keep_mask]
        class_sum = class_count_eff.sum()
        if class_sum == 0.0:
            prior = np.ones_like(class_count_eff, dtype=float) / len(class_count_eff)
        else:
            prior = class_count_eff / class_sum

        labels_arr = np.asarray(self.labels, dtype=object)
        labels = labels_arr[keep_mask].tolist()

        results = FederatedGaussianNBResults(
            x_vars=self.x_vars,
            labels=labels,
            theta=means.astype(float, copy=False),
            var=var.astype(float, copy=False),
            class_count=class_count_eff.astype(float, copy=False),
            class_prior=prior.astype(float, copy=False),
            n_obs=int(total_n_obs),
            var_smoothing=self.var_smoothing,
        )
        self.results = results
        return results


def _ensure_numeric_features(X: np.ndarray, x_vars: List[str]) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X_enc = X[x_vars].to_numpy(dtype=float, copy=False)
    else:
        X_enc = np.asarray(X, dtype=float)

    if X_enc.ndim == 1:
        X_enc = X_enc.reshape(-1, len(x_vars))
    return X_enc


def _ensure_training_data(X, y, x_vars):
    X_enc = _ensure_numeric_features(X, x_vars)
    if y is None:
        raise ValueError("Missing target values for Gaussian naive Bayes.")
    y_arr = np.asarray(y)
    return X_enc, y_arr
