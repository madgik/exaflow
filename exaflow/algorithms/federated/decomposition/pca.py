from __future__ import annotations

from typing import Optional

import numpy as np

from exaflow.algorithms.federated.utils import to_numpy


def global_mean_std(agg_client, X: np.ndarray):
    """Global observation count, column means and sample standard deviations.

    Two aggregation rounds: n and column sums give the global mean, then the
    centered sums of squares give the variance. Raw moments (sum(x^2) - n*mean^2)
    lose all precision when the mean is large relative to the spread.
    """
    totals = np.asarray(
        agg_client.sum(np.append(float(len(X)), np.einsum("ij->j", X))), dtype=float
    ).reshape(-1)
    total_n_obs = float(totals[0])
    means = totals[1:] / total_n_obs

    centered = X - means
    total_ssd = np.asarray(
        agg_client.sum(np.einsum("ij,ij->j", centered, centered)), dtype=float
    ).reshape(-1)
    sigmas = np.sqrt(total_ssd / (total_n_obs - 1))
    return total_n_obs, means, sigmas


class FederatedPCA:
    """Federated PCA via aggregated moments and eigendecomposition."""

    def __init__(self, agg_client, *, copy: bool = False):
        self.agg_client = agg_client
        self.copy = copy

    def fit(self, X, y: Optional[None] = None):
        X = to_numpy(X)

        total_n_obs, means, sigmas = global_mean_std(self.agg_client, X)
        zero_sigma = sigmas == 0
        if np.any(zero_sigma):
            sigmas = sigmas.copy()
            sigmas[zero_sigma] = 1.0

        if self.copy or not X.flags.writeable:
            X = np.array(X, copy=True)

        np.subtract(X, means, out=X)
        np.divide(X, sigmas, out=X)

        gramian = np.einsum("ji,jk->ik", X, X)
        total_gramian = np.asarray(self.agg_client.sum(gramian), dtype=float).reshape(
            gramian.shape
        )
        covariance = total_gramian / (total_n_obs - 1)

        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors.T

        self.n_samples_seen_ = int(total_n_obs)
        self.mean_ = means
        self.scale_ = sigmas
        self.components_ = eigenvectors.real
        self.explained_variance_ = eigenvalues.real
        return self

    def transform(self, X):
        X = to_numpy(X)
        if self.copy or not X.flags.writeable:
            X = np.array(X, copy=True)
        np.subtract(X, self.mean_, out=X)
        np.divide(X, self.scale_, out=X)
        return np.dot(X, self.components_.T)

    def fit_transform(self, X, y: Optional[None] = None):
        self.fit(X, y=y)
        return self.transform(X)
