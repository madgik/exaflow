from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from exaflow.algorithms.federated.cluster.kmeans import INIT_MULTI_START_RANDOM_RANGE
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.utils import BadInputError
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
    def _outputs_equal(self, left, right):
        if isinstance(left, FederatedKMeans) and isinstance(right, FederatedKMeans):
            left_vars = {
                key: value
                for key, value in vars(left).items()
                if key not in {"agg_client", "labels_"}
            }
            right_vars = {
                key: value
                for key, value in vars(right).items()
                if key not in {"agg_client", "labels_"}
            }
            return super()._outputs_equal(left_vars, right_vars)
        return super()._outputs_equal(left, right)

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


def test_kmeans_exposes_internal_fit_state_for_reporting_and_preprocessing():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ]
    )
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)

    model = FederatedKMeans(
        agg_client=agg_client,
        n_clusters=2,
        random_state=3,
        maxiter=100,
        tol=1e-8,
    ).fit(X, feature_names=["age", "crp"])

    assert model.n_obs_ == 4
    assert model.n_features_ == 2
    assert model.feature_names_ == ["age", "crp"]
    assert len(model.cluster_counts_) == 2
    assert sum(model.cluster_counts_) == 4
    assert model.labels_.shape == (4,)
    assert model.predict(X).tolist() == model.labels_.tolist()
    assert len(model.cluster_inertia_) == 2
    assert model.inertia_ == pytest.approx(sum(model.cluster_inertia_))
    assert model.n_iter_ >= 1
    assert isinstance(model.converged_, bool)
    assert model.empty_clusters_ == []
    assert model.init_method_ == "random_range"
    assert model.n_init_ == 1
    assert model.best_init_ == 0
    assert model.random_state_ == 3


def test_kmeans_tracks_empty_clusters_without_exposing_fake_count():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)

    model = FederatedKMeans(
        agg_client=agg_client,
        n_clusters=4,
        random_state=6,
        maxiter=10,
    ).fit(X)

    assert len(model.cluster_counts_) == 4
    assert sum(model.cluster_counts_) == 2
    assert model.empty_clusters_
    assert all(model.cluster_counts_[idx] == 0 for idx in model.empty_clusters_)


def test_kmeans_final_counts_match_final_labels_after_last_update():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [4.8, 5.0],
            [5.0, 5.1],
            [9.9, 10.0],
        ]
    )
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)

    model = FederatedKMeans(
        agg_client=agg_client,
        n_clusters=3,
        random_state=12,
        maxiter=1,
        tol=0.0,
    ).fit(X)

    assert (
        model.cluster_counts_
        == np.bincount(
            model.labels_,
            minlength=model.n_clusters,
        ).tolist()
    )
    centers = np.asarray(model.cluster_centers_, dtype=float)
    for cluster_idx, count in enumerate(model.cluster_counts_):
        if count == 0:
            assert np.allclose(centers[cluster_idx], 0.0)
            continue
        expected_center = X[model.labels_ == cluster_idx].mean(axis=0)
        assert centers[cluster_idx] == pytest.approx(expected_center)


def test_kmeans_fit_predict_returns_local_labels():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    model = FederatedKMeans(
        agg_client=agg_client,
        n_clusters=2,
        random_state=5,
    )

    labels = model.fit_predict(X)

    assert labels.tolist() == model.labels_.tolist()
    assert labels.shape == (4,)


def test_kmeans_rejects_non_finite_values():
    X = np.array([[0.0, 1.0], [np.nan, 2.0]])
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    model = FederatedKMeans(
        agg_client=agg_client,
        n_clusters=2,
        random_state=5,
    )

    with pytest.raises(BadInputError, match="requires finite numerical values"):
        model.fit(X)


def test_kmeans_coordinates_non_finite_validation_across_workers():
    coordinator = AggregationCoordinator(n_workers=3)
    partitions = [
        np.array([[0.0, 1.0], [0.1, 1.1]]),
        np.array([[np.nan, 2.0]]),
        np.array([[5.0, 5.0], [5.1, 5.1]]),
    ]

    def fit_partition(worker_id):
        agg_client = SimulatedAggClient(
            worker_id=worker_id,
            coordinator=coordinator,
        )
        with pytest.raises(BadInputError, match="requires finite numerical values"):
            FederatedKMeans(
                agg_client=agg_client,
                n_clusters=2,
                random_state=5,
            ).fit(partitions[worker_id])

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fit_partition, worker_id) for worker_id in range(3)]
        for future in futures:
            future.result(timeout=2)


@pytest.mark.parametrize(
    "params, error",
    [
        ({"n_clusters": 0}, "n_clusters"),
        ({"n_clusters": 2, "maxiter": 0}, "maxiter"),
        ({"n_clusters": 2, "n_init": 0}, "n_init"),
        ({"n_clusters": 2, "init_method": "kmeans++"}, "initialization method"),
    ],
)
def test_kmeans_rejects_invalid_hyperparameters(params, error):
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    model = FederatedKMeans(
        agg_client=agg_client,
        **params,
    )

    with pytest.raises(BadInputError, match=error):
        model.fit(np.array([[0.0], [1.0]]))


def test_kmeans_multi_start_keeps_lowest_inertia_initialization():
    X = np.vstack(
        [
            np.tile(np.array([0.0, 0.0]), (5, 1)),
            np.tile(np.array([5.0, 5.0]), (5, 1)),
            np.tile(np.array([10.0, 10.0]), (5, 1)),
        ]
    )
    random_state = 11
    n_init = 4
    single_start_inertias = []
    for init_idx in range(n_init):
        coordinator = AggregationCoordinator(n_workers=1)
        agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
        model = FederatedKMeans(
            agg_client=agg_client,
            n_clusters=3,
            random_state=random_state + init_idx,
            maxiter=100,
            tol=1e-8,
        ).fit(X)
        single_start_inertias.append(model.inertia_)

    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    multi_start = FederatedKMeans(
        agg_client=agg_client,
        n_clusters=3,
        init_method=INIT_MULTI_START_RANDOM_RANGE,
        n_init=n_init,
        random_state=random_state,
        maxiter=100,
        tol=1e-8,
    ).fit(X)

    assert multi_start.init_method_ == INIT_MULTI_START_RANDOM_RANGE
    assert multi_start.n_init_ == n_init
    assert multi_start.inertia_ == pytest.approx(min(single_start_inertias))
    assert multi_start.best_init_ == int(np.argmin(single_start_inertias))
