from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.utils import _to_numpy


class FederatedKMeans:
    """
    Federated K-means estimator exposing a sklearn-like `fit` interface.

    The implementation gathers distributed min/max to initialize centers,
    executes Lloyd iterations via aggregation of sums/counts,
    and resets empty clusters to the origin until the Frobenius norm between center
    updates is below `tol`.
    """

    def __init__(
        self,
        agg_client,
        *,
        n_clusters,
        tol=1e-4,
        maxiter=100,
        random_state=123,
    ):
        self.agg_client = agg_client
        self.n_clusters = int(n_clusters)
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.random_state = int(random_state)

    def fit(self, x):
        X = _to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_local, n_features = X.shape

        # Global number of observations
        total_n_obs = int(self.agg_client.sum([float(n_local)])[0])

        # If there is no data at all, return empty centers
        if total_n_obs == 0:
            self.n_obs_ = 0
            self.cluster_centers_ = []
            return self

        if n_local > 0:
            local_min = np.nanmin(X, axis=0)
            local_max = np.nanmax(X, axis=0)
        else:
            local_min = np.full((n_features,), np.inf, dtype=float)
            local_max = np.full((n_features,), -np.inf, dtype=float)

        global_min = np.asarray(self.agg_client.min(local_min), dtype=float)
        global_max = np.asarray(self.agg_client.max(local_max), dtype=float)

        rng = np.random.RandomState(seed=self.random_state)
        centers = rng.uniform(
            low=global_min,
            high=global_max,
            size=(int(self.n_clusters), n_features),
        )

        for _ in range(int(self.maxiter)):
            if n_local > 0:
                diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists_sq = np.einsum("ijk,ijk->ij", diff, diff)
                labels = np.argmin(dists_sq, axis=1)

                sum_local = np.zeros((self.n_clusters, n_features), dtype=float)
                count_local = np.zeros((self.n_clusters,), dtype=float)

                for k in range(self.n_clusters):
                    mask = labels == k
                    if np.any(mask):
                        sum_local[k] = X[mask].sum(axis=0)
                        count_local[k] = float(mask.sum())
            else:
                sum_local = np.zeros((self.n_clusters, n_features), dtype=float)
                count_local = np.zeros((self.n_clusters,), dtype=float)

            sum_global_arr = self.agg_client.sum(sum_local.ravel())
            count_global_arr = self.agg_client.sum(count_local)
            sum_global = np.asarray(sum_global_arr, dtype=float).reshape(
                (self.n_clusters, n_features)
            )
            count_global = np.asarray(count_global_arr, dtype=float)

            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                if count_global[k] > 0.0:
                    new_centers[k] = sum_global[k] / count_global[k]
                else:
                    new_centers[k] = np.zeros(n_features, dtype=float)

            diff_norm = np.linalg.norm(new_centers - centers, ord="fro")
            centers = new_centers
            if diff_norm <= self.tol:
                break

        self.n_obs_ = int(total_n_obs)
        self.cluster_centers_ = [
            [float(value) for value in center] for center in centers
        ]
        return self
