from __future__ import annotations

from typing import Optional

import numpy as np

from exaflow.algorithms.federated.utils import to_numpy
from exaflow.algorithms.federated.validation import BOOLEAN_RULE
from exaflow.algorithms.federated.validation import ArrayPolicy
from exaflow.algorithms.federated.validation import check_array
from exaflow.algorithms.federated.validation import check_feature_schema
from exaflow.algorithms.federated.validation import check_fitted
from exaflow.algorithms.federated.validation import check_sample_counts
from exaflow.algorithms.federated.validation import validate_parameters

SIGN_ANCHOR_RTOL = 1e-12
SIGN_ANCHOR_ATOL = 1e-15


def canonicalize_component_signs(components):
    """Choose a deterministic sign for each PCA component.

    Eigenvectors are defined up to sign.  Choosing the loading with the
    greatest absolute magnitude as the anchor makes component replay stable
    for well-separated eigenvalues.  Loadings within numerical tolerance of
    the maximum are treated as tied, and the first tied feature is selected.
    """
    components = np.asarray(components, dtype=float)
    if components.ndim != 2:
        raise ValueError("components must be a two-dimensional array.")

    canonicalized = np.array(components, copy=True)
    absolute_loadings = np.abs(canonicalized)
    maximum_loadings = absolute_loadings.max(axis=1, keepdims=True)
    tied_anchors = np.isclose(
        absolute_loadings,
        maximum_loadings,
        rtol=SIGN_ANCHOR_RTOL,
        atol=SIGN_ANCHOR_ATOL,
    )
    anchors = np.argmax(tied_anchors, axis=1)
    signs = np.sign(canonicalized[np.arange(canonicalized.shape[0]), anchors])
    signs[signs == 0] = 1.0
    canonicalized *= signs[:, np.newaxis]
    return canonicalized


class FederatedPCA:
    """Federated PCA via aggregated moments and eigendecomposition."""

    def __init__(self, agg_client, *, copy: bool = False):
        self.agg_client = agg_client
        self.copy = copy

    def fit(self, X, y: Optional[None] = None):
        self._fit(X)
        return self

    def _fit(self, X):
        """Fit the model and return the standardized local working array."""
        validate_parameters({"copy": self.copy}, {"copy": BOOLEAN_RULE})
        X = check_array(to_numpy(X), policy=ArrayPolicy(), input_name="Features")

        n_obs = len(X)
        sx = np.einsum("ij->j", X)
        sxx = np.einsum("ij,ij->j", X, X)

        total_n_obs_arr = self.agg_client.sum(np.array([float(n_obs)], dtype=float))
        total_sx = self.agg_client.sum(sx)
        total_sxx = self.agg_client.sum(sxx)

        total_n_obs = float(total_n_obs_arr.reshape(-1)[0])
        total_sx = np.asarray(total_sx, dtype=float)
        total_sxx = np.asarray(total_sxx, dtype=float)

        check_sample_counts(total_n_obs, minimum=2, name="PCA")
        means = total_sx / total_n_obs
        variances = (total_sxx - total_n_obs * means**2) / (total_n_obs - 1)
        variances = np.maximum(variances, 0.0)
        sigmas = np.sqrt(variances)
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
        covariance = (covariance + covariance.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors.T

        self.n_samples_seen_ = int(total_n_obs)
        self.mean_ = means
        self.scale_ = sigmas
        self.components_ = canonicalize_component_signs(eigenvectors.real)
        self.explained_variance_ = eigenvalues.real
        return X

    def transform(self, X):
        check_fitted(self, "mean_", "scale_", "components_")
        X = check_array(
            to_numpy(X), policy=ArrayPolicy(dimensions=(1, 2)), input_name="Features"
        )
        check_feature_schema(X, n_features=self.mean_.size)
        if self.copy or not X.flags.writeable:
            X = np.array(X, copy=True)
        np.subtract(X, self.mean_, out=X)
        np.divide(X, self.scale_, out=X)
        return np.dot(X, self.components_.T)

    def fit_transform(self, X, y: Optional[None] = None):
        standardized_X = self._fit(X)
        return np.dot(standardized_X, self.components_.T)
