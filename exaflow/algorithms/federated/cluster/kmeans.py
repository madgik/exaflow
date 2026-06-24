from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import to_numpy

INIT_RANDOM_RANGE = "random_range"
INIT_MULTI_START_RANDOM_RANGE = "multi_start_random_range"


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
        init_method="random_range",
        n_init=1,
        tol=1e-4,
        maxiter=100,
        random_state=123,
    ):
        self.agg_client = agg_client
        self.n_clusters = int(n_clusters)
        self.init_method = str(init_method)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.random_state = int(random_state)

    def fit(self, x, *, feature_names=None):
        self._validate_hyperparameters()
        X = to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if not np.all(np.isfinite(X)):
            raise BadInputError(
                "K-means requires finite numerical values. Apply missing-value "
                "handling and remove infinite values before fitting."
            )

        n_local, n_features = X.shape
        self.n_features_ = int(n_features)
        self.feature_names_ = list(feature_names) if feature_names is not None else None
        self.init_method_ = self.init_method
        self.n_init_ = self._effective_n_init()
        self.random_state_ = self.random_state

        # Global number of observations
        total_n_obs = int(self.agg_client.sum([float(n_local)])[0])

        # If there is no data at all, return empty centers
        if total_n_obs == 0:
            self.n_obs_ = 0
            self.cluster_centers_ = []
            self.cluster_counts_ = []
            self.labels_ = np.asarray([], dtype=int)
            self.inertia_ = 0.0
            self.cluster_inertia_ = []
            self.n_iter_ = 0
            self.converged_ = True
            self.empty_clusters_ = []
            self.best_init_ = 0
            return self

        if n_local > 0:
            local_min = np.nanmin(X, axis=0)
            local_max = np.nanmax(X, axis=0)
        else:
            local_min = np.full((n_features,), np.inf, dtype=float)
            local_max = np.full((n_features,), -np.inf, dtype=float)

        global_min = np.asarray(self.agg_client.min(local_min), dtype=float)
        global_max = np.asarray(self.agg_client.max(local_max), dtype=float)

        best_result = None
        for init_idx in range(self.n_init_):
            centers = self._initialize_centers(
                global_min=global_min,
                global_max=global_max,
                n_features=n_features,
                init_idx=init_idx,
            )
            result = self._fit_one_initialization(
                X=X,
                centers=centers,
                n_local=n_local,
                n_features=n_features,
                init_idx=init_idx,
            )
            if best_result is None or result["inertia"] < best_result["inertia"]:
                best_result = result

        centers = best_result["centers"]
        labels = best_result["labels"]
        count_global = best_result["counts"]
        cluster_inertia_global = best_result["cluster_inertia"]

        self.n_obs_ = int(total_n_obs)
        self.cluster_centers_ = [
            [float(value) for value in center] for center in centers
        ]
        self.cluster_counts_ = [int(value) for value in count_global]
        self.labels_ = labels.astype(int, copy=False)
        self.inertia_ = float(cluster_inertia_global.sum())
        self.cluster_inertia_ = [float(value) for value in cluster_inertia_global]
        self.n_iter_ = int(best_result["n_iter"])
        self.converged_ = bool(best_result["converged"])
        self.empty_clusters_ = [
            int(k) for k, count in enumerate(count_global) if count <= 0.0
        ]
        self.best_init_ = int(best_result["init_idx"])
        return self

    def predict(self, x):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("FederatedKMeans instance is not fitted yet.")
        X = to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        centers = np.asarray(self.cluster_centers_, dtype=float)
        if centers.size == 0:
            return np.asarray([], dtype=int)
        return self._assign_labels(X, centers).astype(int, copy=False)

    def fit_predict(self, x, *, feature_names=None):
        self.fit(x, feature_names=feature_names)
        return self.labels_

    def _fit_one_initialization(
        self,
        *,
        X,
        centers,
        n_local,
        n_features,
        init_idx,
    ):
        labels = np.asarray([], dtype=int)
        count_global = np.zeros((self.n_clusters,), dtype=float)
        converged = False
        n_iter = 0
        for iteration in range(int(self.maxiter)):
            n_iter = iteration + 1
            if n_local > 0:
                labels = self._assign_labels(X, centers)
                sum_local, count_local = self._compute_local_sums_and_counts(
                    X=X,
                    labels=labels,
                    n_features=n_features,
                )
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
                converged = True
                break

        labels = self._assign_labels(X, centers) if n_local > 0 else labels
        _, final_count_local = self._compute_local_sums_and_counts(
            X=X,
            labels=labels,
            n_features=n_features,
        )
        final_count_global_arr = self.agg_client.sum(final_count_local)
        final_count_global = np.asarray(final_count_global_arr, dtype=float)
        cluster_inertia_local = self._compute_cluster_inertia(
            X=X,
            centers=centers,
            labels=labels,
        )
        cluster_inertia_global_arr = self.agg_client.sum(cluster_inertia_local)
        cluster_inertia_global = np.asarray(cluster_inertia_global_arr, dtype=float)
        return {
            "centers": centers,
            "labels": labels,
            "counts": final_count_global,
            "cluster_inertia": cluster_inertia_global,
            "inertia": float(cluster_inertia_global.sum()),
            "n_iter": n_iter,
            "converged": converged,
            "init_idx": int(init_idx),
        }

    def _initialize_centers(self, *, global_min, global_max, n_features, init_idx):
        if self.init_method not in {
            INIT_RANDOM_RANGE,
            INIT_MULTI_START_RANDOM_RANGE,
        }:
            raise ValueError(
                f"Unsupported KMeans initialization method: '{self.init_method}'."
            )
        rng = np.random.RandomState(seed=self.random_state + int(init_idx))
        return rng.uniform(
            low=global_min,
            high=global_max,
            size=(int(self.n_clusters), n_features),
        )

    def _effective_n_init(self):
        if self.n_init < 1:
            raise BadInputError("n_init must be greater than or equal to 1.")
        if self.init_method == INIT_RANDOM_RANGE:
            return 1
        if self.init_method == INIT_MULTI_START_RANDOM_RANGE:
            return self.n_init
        raise BadInputError(
            f"Unsupported KMeans initialization method: '{self.init_method}'."
        )

    def _validate_hyperparameters(self):
        if self.n_clusters < 1:
            raise BadInputError("n_clusters must be greater than or equal to 1.")
        if self.maxiter < 1:
            raise BadInputError("maxiter must be greater than or equal to 1.")
        self._effective_n_init()

    @staticmethod
    def _assign_labels(X, centers):
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.einsum("ijk,ijk->ij", diff, diff)
        return np.argmin(dists_sq, axis=1)

    def _compute_local_sums_and_counts(self, *, X, labels, n_features):
        sum_local = np.zeros((self.n_clusters, n_features), dtype=float)
        count_local = np.zeros((self.n_clusters,), dtype=float)
        if X.shape[0] == 0:
            return sum_local, count_local
        np.add.at(sum_local, labels, X)
        count_local = np.bincount(labels, minlength=self.n_clusters).astype(float)
        return sum_local, count_local

    def _compute_cluster_inertia(self, *, X, centers, labels):
        cluster_inertia = np.zeros((self.n_clusters,), dtype=float)
        if X.shape[0] == 0:
            return cluster_inertia
        for k in range(self.n_clusters):
            mask = labels == k
            if not np.any(mask):
                continue
            diff = X[mask] - centers[k]
            cluster_inertia[k] = float(np.einsum("ij,ij->", diff, diff))
        return cluster_inertia
