import numpy as np
import pytest

from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from tests.standalone_tests.federated_algorithms.utils import FederatedAlgorithmTest
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)

TEST_CASES = [
    (
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        2,
        0,
        "n_obs",
        None,
    ),
    (
        np.random.RandomState(1).randn(15, 2),
        3,
        1,
        "centers_shape",
        (3, 2),
    ),
    (
        np.array([[1.0, 2.0], [1.2, 1.8], [0.9, 2.1]]),
        1,
        2,
        "center_near_mean",
        None,
    ),
    (
        np.vstack(
            [
                np.tile(np.array([0.0, 0.0]), (5, 1)),
                np.tile(np.array([5.0, 0.0]), (5, 1)),
            ]
        ),
        2,
        3,
        "two_cluster_means",
        None,
    ),
    (
        np.zeros((0, 3)),
        2,
        4,
        "empty",
        None,
    ),
    (
        np.array([[0.0], [0.1], [5.0], [5.1]]),
        2,
        5,
        "single_feature",
        (2, 1),
    ),
    (
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        4,
        6,
        "empty_cluster_zero",
        None,
    ),
    (
        np.random.RandomState(8).randn(20, 8),
        3,
        8,
        "high_dim",
        (3, 8),
    ),
]


class TestFederatedKMeans(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        n_clusters = kwargs["n_clusters"]
        random_state = kwargs["random_state"]
        agg_client = kwargs["centralized_agg_client"]
        model = FederatedKMeans(
            agg_client=agg_client,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        model.fit(X)
        return model

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        n_clusters = kwargs["n_clusters"]
        random_state = kwargs["random_state"]
        model = FederatedKMeans(
            agg_client=agg_client,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        model.fit(X)
        return model

    def compare(self, federated_output, centralized_output, **kwargs):
        check = kwargs["check"]
        expected = kwargs.get("expected")
        X = kwargs["X_full"]

        if check == "n_obs":
            assert federated_output.n_obs_ == X.shape[0]
        elif check == "centers_shape":
            centers = np.asarray(federated_output.cluster_centers_)
            assert centers.shape == expected
        elif check == "center_near_mean":
            center = np.asarray(federated_output.cluster_centers_)[0]
            assert np.allclose(center, np.mean(X, axis=0), atol=1e-6)
        elif check == "two_cluster_means":
            centers = np.asarray(federated_output.cluster_centers_)
            assert any(
                np.linalg.norm(center - np.array([0.0, 0.0])) < 1e-6
                for center in centers
            )
            assert any(
                np.linalg.norm(center - np.array([5.0, 0.0])) < 1e-6
                for center in centers
            )
        elif check == "empty":
            assert federated_output.n_obs_ == 0
            assert federated_output.cluster_centers_ == []
        elif check == "single_feature":
            centers = np.asarray(federated_output.cluster_centers_)
            assert centers.shape == expected
        elif check == "empty_cluster_zero":
            centers = np.asarray(federated_output.cluster_centers_)
            assert any(np.allclose(center, 0.0, atol=1e-8) for center in centers)
        elif check == "high_dim":
            centers = np.asarray(federated_output.cluster_centers_)
            assert centers.shape == expected
            assert federated_output.n_obs_ == X.shape[0]
        else:
            raise ValueError(f"Unknown check: {check}")

    @pytest.mark.parametrize("X, n_clusters, random_state, check, expected", TEST_CASES)
    def test_federated_algorithm_with_one_worker(
        self, X, n_clusters, random_state, check, expected
    ):
        coordinator = AggregationCoordinator(n_workers=1)
        centralized_agg_client = SimulatedAggClient(
            worker_id=0, coordinator=coordinator
        )
        self.run_comparison(
            X=X,
            y=np.zeros((X.shape[0],), dtype=float),
            n_workers=1,
            n_clusters=n_clusters,
            random_state=random_state,
            check=check,
            expected=expected,
            X_full=X,
            centralized_agg_client=centralized_agg_client,
        )

    @pytest.mark.parametrize("X, n_clusters, random_state, check, expected", TEST_CASES)
    def test_federated_algorithm_with_multiple_workers(
        self, X, n_clusters, random_state, check, expected
    ):
        coordinator = AggregationCoordinator(n_workers=1)
        centralized_agg_client = SimulatedAggClient(
            worker_id=0, coordinator=coordinator
        )
        self.run_comparison(
            X=X,
            y=np.zeros((X.shape[0],), dtype=float),
            n_workers=3,
            n_clusters=n_clusters,
            random_state=random_state,
            check=check,
            expected=expected,
            X_full=X,
            centralized_agg_client=centralized_agg_client,
        )
