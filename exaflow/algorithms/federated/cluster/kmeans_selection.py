from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.utils import BadInputError


class FederatedKMeansSelector:
    """Fit KMeans over a k range and select k using a simple elbow heuristic."""

    def __init__(
        self,
        agg_client,
        *,
        k_min,
        k_max,
        init_method="random_range",
        n_init=1,
        tol=1e-4,
        maxiter=100,
        random_state=123,
    ):
        self.agg_client = agg_client
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.init_method = str(init_method)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.random_state = int(random_state)

    def fit(self, x, *, feature_names=None):
        if self.k_min < 1:
            raise BadInputError("k_min must be greater than or equal to 1.")
        if self.k_max < self.k_min:
            raise BadInputError("k_max must be greater than or equal to k_min.")

        self.models_by_k_ = {}
        self.inertia_by_k_ = {}
        for k in range(self.k_min, self.k_max + 1):
            model = FederatedKMeans(
                agg_client=self.agg_client,
                n_clusters=k,
                init_method=self.init_method,
                n_init=self.n_init,
                tol=self.tol,
                maxiter=self.maxiter,
                random_state=self.random_state,
            ).fit(x, feature_names=feature_names)
            self.models_by_k_[k] = model
            self.inertia_by_k_[k] = float(model.inertia_)

        self.selected_k_, self.warning_ = self._select_k(self.inertia_by_k_)
        self.best_model_ = self.models_by_k_[self.selected_k_]
        return self

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
