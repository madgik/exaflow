from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import to_numpy
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimator

INIT_RANDOM_RANGE = "random_range"
INIT_MULTI_START_RANDOM_RANGE = "multi_start_random_range"


def assign_clusters(x, centers):
    """Assign observations to the nearest supplied centers."""
    X = to_numpy(x)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[0] == 0:
        raise BadInputError("At least one two-dimensional center is required.")
    if X.shape[1] != centers.shape[1]:
        raise BadInputError("Observations and centers must have matching features.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(centers)):
        raise BadInputError("K-means requires finite numerical values.")
    return FederatedKMeans._assign_labels(X, centers).astype(int, copy=False)


@dataclass
class FederatedKMeansResults:
    n_obs_: int
    n_features_: int
    feature_names_: Optional[list[str]]
    n_clusters: int
    cluster_centers_: list[list[float]]
    cluster_counts_: list[int]
    labels_: np.ndarray
    inertia_: float
    cluster_inertia_: list[float]
    n_iter_: int
    converged_: bool
    empty_clusters_: list[int]
    init_method_: str
    n_init_: int
    best_init_: int
    random_state_: int

    @property
    def nobs(self) -> int:
        return self.n_obs_

    def predict(self, x):
        if not self.cluster_centers_:
            return np.asarray([], dtype=int)
        return assign_clusters(x, self.cluster_centers_)


@dataclass
class FederatedKMeansSelectionResults:
    models_by_k: dict[int, FederatedKMeansResults]
    inertia_by_k: dict[int, float]
    selected_k: int
    best_model: FederatedKMeansResults
    warning: Optional[str]


class FederatedKMeans(FederatedEstimator):
    """
    Federated K-means estimator exposing a sklearn-like `fit` interface.

    The implementation gathers distributed min/max to initialize centers,
    executes Lloyd iterations via aggregation of sums/counts,
    and resets empty clusters to the origin until the Frobenius norm between center
    updates is below `tol`.
    """

    def __init__(
        self,
        *,
        n_clusters,
        init_method="random_range",
        n_init=1,
        tol=1e-4,
        maxiter=100,
        random_state=123,
    ):
        self.n_clusters = int(n_clusters)
        self.init_method = str(init_method)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.random_state = int(random_state)

    def fit(self, x, y=None, *, agg_client, feature_names=None):
        del y
        self._agg_client = agg_client
        self._validate_hyperparameters()
        X = to_numpy(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        local_invalid = 0.0 if np.all(np.isfinite(X)) else 1.0
        global_invalid = float(self._agg_client.sum([local_invalid])[0])
        if global_invalid > 0.0:
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
        total_n_obs = int(self._agg_client.sum([float(n_local)])[0])

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
            return self._build_results_without_aggregation_client()

        if n_local > 0:
            local_min = np.nanmin(X, axis=0)
            local_max = np.nanmax(X, axis=0)
        else:
            local_min = np.full((n_features,), np.inf, dtype=float)
            local_max = np.full((n_features,), -np.inf, dtype=float)

        global_min = np.asarray(self._agg_client.min(local_min), dtype=float)
        global_max = np.asarray(self._agg_client.max(local_max), dtype=float)

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
        return self._build_results_without_aggregation_client()

    @classmethod
    def select_k(
        cls,
        x,
        *,
        k_min,
        k_max,
        agg_client,
        init_method="random_range",
        n_init=1,
        tol=1e-4,
        maxiter=100,
        random_state=123,
        feature_names=None,
    ):
        k_min = int(k_min)
        k_max = int(k_max)
        if k_min < 1:
            raise BadInputError("k_min must be greater than or equal to 1.")
        if k_max < k_min:
            raise BadInputError("k_max must be greater than or equal to k_min.")

        models_by_k = {}
        inertia_by_k = {}
        for n_clusters in range(k_min, k_max + 1):
            model = cls(
                n_clusters=n_clusters,
                init_method=init_method,
                n_init=n_init,
                tol=tol,
                maxiter=maxiter,
                random_state=random_state,
            ).fit(x, agg_client=agg_client, feature_names=feature_names)
            models_by_k[n_clusters] = model
            inertia_by_k[n_clusters] = float(model.inertia_)

        selected_k, warning = cls._select_k(inertia_by_k)
        return FederatedKMeansSelectionResults(
            models_by_k=models_by_k,
            inertia_by_k=inertia_by_k,
            selected_k=selected_k,
            best_model=models_by_k[selected_k],
            warning=warning,
        )

    def _build_results(self):
        return FederatedKMeansResults(
            n_obs_=self.n_obs_,
            n_features_=self.n_features_,
            feature_names_=self.feature_names_,
            n_clusters=self.n_clusters,
            cluster_centers_=self.cluster_centers_,
            cluster_counts_=self.cluster_counts_,
            labels_=self.labels_,
            inertia_=self.inertia_,
            cluster_inertia_=self.cluster_inertia_,
            n_iter_=self.n_iter_,
            converged_=self.converged_,
            empty_clusters_=self.empty_clusters_,
            init_method_=self.init_method_,
            n_init_=self.n_init_,
            best_init_=self.best_init_,
            random_state_=self.random_state_,
        )

    def _build_results_without_aggregation_client(self):
        results = self._build_results()
        del self._agg_client
        return results

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

            sum_global_arr = self._agg_client.sum(sum_local.ravel())
            count_global_arr = self._agg_client.sum(count_local)
            sum_global = np.asarray(sum_global_arr, dtype=float).reshape(
                (self.n_clusters, n_features)
            )
            count_global = np.asarray(count_global_arr, dtype=float)

            new_centers = self._compute_centers_from_sums_and_counts(
                sums=sum_global,
                counts=count_global,
                n_features=n_features,
                like=centers,
            )

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
        final_count_global_arr = self._agg_client.sum(final_count_local)
        final_count_global = np.asarray(final_count_global_arr, dtype=float)
        cluster_inertia_local = self._compute_cluster_inertia(
            X=X,
            centers=centers,
            labels=labels,
        )
        cluster_inertia_global_arr = self._agg_client.sum(cluster_inertia_local)
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
    def _select_k(inertia_by_k):
        k_values = np.asarray(list(inertia_by_k.keys()), dtype=float)
        inertia_values = np.asarray(list(inertia_by_k.values()), dtype=float)
        if len(k_values) == 1:
            return int(k_values[0]), "Only one k value was evaluated."
        if len(k_values) == 2:
            return int(k_values[0]), "Elbow selection is ambiguous with two k values."

        first = np.array([k_values[0], inertia_values[0]], dtype=float)
        last = np.array([k_values[-1], inertia_values[-1]], dtype=float)
        line = last - first
        line_norm = np.linalg.norm(line)
        if line_norm == 0.0:
            return int(k_values[0]), "Elbow selection is ambiguous."

        distances = []
        for k_value, inertia in zip(k_values, inertia_values):
            point = np.array([k_value, inertia], dtype=float)
            offset = first - point
            distance = abs(line[0] * offset[1] - line[1] * offset[0]) / line_norm
            distances.append(float(distance))

        max_distance_idx = int(np.argmax(distances))
        warning = None
        if distances[max_distance_idx] <= 1e-12:
            warning = "Elbow selection is ambiguous."
        return int(k_values[max_distance_idx]), warning

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

    def _compute_centers_from_sums_and_counts(self, *, sums, counts, n_features, like):
        centers = np.zeros_like(like)
        for k in range(self.n_clusters):
            if counts[k] > 0.0:
                centers[k] = sums[k] / counts[k]
            else:
                centers[k] = np.zeros(n_features, dtype=float)
        return centers

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
