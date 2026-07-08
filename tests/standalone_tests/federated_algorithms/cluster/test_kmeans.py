from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from exaflow.algorithms.federated.cluster.kmeans import INIT_MULTI_START_RANDOM_RANGE
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeansResults
from exaflow.algorithms.federated.cluster.kmeans import assign_clusters
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
        if isinstance(left, FederatedKMeansResults) and isinstance(
            right, FederatedKMeansResults
        ):
            left_vars = {
                key: value for key, value in vars(left).items() if key != "labels_"
            }
            right_vars = {
                key: value for key, value in vars(right).items() if key != "labels_"
            }
            return super()._outputs_equal(left_vars, right_vars)
        return super()._outputs_equal(left, right)

    def compute_centralized_result(self, X, y, **kwargs):
        n_clusters = kwargs["n_clusters"]
        random_state = kwargs["random_state"]
        agg_client = kwargs["centralized_agg_client"]
        model = FederatedKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
        )
        return model.fit(X, agg_client=agg_client)

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        n_clusters = kwargs["n_clusters"]
        random_state = kwargs["random_state"]
        model = FederatedKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
        )
        return model.fit(X, agg_client=agg_client)

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
            n_workers=5,
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

    result = FederatedKMeans(
        n_clusters=2,
        random_state=3,
        maxiter=100,
        tol=1e-8,
    ).fit(X, agg_client=agg_client, feature_names=["age", "crp"])

    assert result.nobs == 4
    assert result.n_features_ == 2
    assert result.feature_names_ == ["age", "crp"]
    assert len(result.cluster_counts_) == 2
    assert sum(result.cluster_counts_) == 4
    assert result.labels_.shape == (4,)
    assert result.predict(X).tolist() == result.labels_.tolist()
    assert len(result.cluster_inertia_) == 2
    assert result.inertia_ == pytest.approx(sum(result.cluster_inertia_))
    assert result.n_iter_ >= 1
    assert isinstance(result.converged_, bool)
    assert result.empty_clusters_ == []
    assert result.init_method_ == "random_range"
    assert result.n_init_ == 1
    assert result.best_init_ == 0
    assert result.random_state_ == 3


def test_kmeans_final_statistics_match_returned_centers_when_not_converged():
    X = np.random.RandomState(1).uniform(0, 10, (12, 1))
    model = FederatedKMeans(
        n_clusters=3,
        maxiter=1,
        tol=0,
        random_state=123,
    )
    result = model.fit(X, agg_client=_agg_client())

    initial_centers = model._initialize_centers(
        global_min=X.min(axis=0),
        global_max=X.max(axis=0),
        n_features=X.shape[1],
        init_idx=0,
    )
    initial_labels = assign_clusters(X, initial_centers)
    expected_centers = np.zeros_like(initial_centers)
    for cluster_idx in range(model.n_clusters):
        if np.any(initial_labels == cluster_idx):
            expected_centers[cluster_idx] = X[initial_labels == cluster_idx].mean(
                axis=0
            )

    expected_labels = assign_clusters(X, result.cluster_centers_)
    centers = np.asarray(result.cluster_centers_, dtype=float)
    expected_inertia = np.sum((X - centers[expected_labels]) ** 2)

    np.testing.assert_allclose(result.cluster_centers_, expected_centers)
    np.testing.assert_array_equal(result.labels_, expected_labels)
    np.testing.assert_allclose(result.inertia_, expected_inertia)
    np.testing.assert_allclose(
        result.inertia_,
        np.sum((X - centers[result.labels_]) ** 2),
    )


def test_kmeans_tracks_empty_clusters_without_exposing_fake_count():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)

    model = FederatedKMeans(
        n_clusters=4,
        random_state=6,
        maxiter=10,
    ).fit(X, agg_client=agg_client)

    assert len(model.cluster_counts_) == 4
    assert sum(model.cluster_counts_) == 2
    assert model.empty_clusters_
    assert all(model.cluster_counts_[idx] == 0 for idx in model.empty_clusters_)


def test_empty_kmeans_results_predict_no_labels():
    agg_client = _agg_client()
    result = FederatedKMeans(n_clusters=2).fit(
        np.empty((0, 2)),
        agg_client=agg_client,
    )

    assert result.predict(np.empty((0, 2))).tolist() == []


def test_kmeans_final_counts_match_final_labels_after_last_iteration():
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
        n_clusters=3,
        random_state=12,
        maxiter=1,
        tol=0.0,
    ).fit(X, agg_client=agg_client)

    assert (
        model.cluster_counts_
        == np.bincount(
            model.labels_,
            minlength=model.n_clusters,
        ).tolist()
    )
    assert model.predict(X).tolist() == model.labels_.tolist()
    assert model.n_iter_ == 1


def test_kmeans_fit_returns_results_with_local_labels():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    result = FederatedKMeans(
        n_clusters=2,
        random_state=5,
    ).fit(X, agg_client=agg_client)

    assert result.labels_.tolist() == result.predict(X).tolist()
    assert result.labels_.shape == (4,)


def test_kmeans_assign_clusters_replays_supplied_centers_deterministically():
    X = np.array([[0.0, 1.0], [9.0, 10.0], [5.0, 5.0]])
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])

    first = assign_clusters(X, centers)
    second = assign_clusters(X, centers)

    np.testing.assert_array_equal(first, [0, 1, 0])
    np.testing.assert_array_equal(first, second)


def test_kmeans_rejects_non_finite_values():
    X = np.array([[0.0, 1.0], [np.nan, 2.0]])
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    model = FederatedKMeans(
        n_clusters=2,
        random_state=5,
    )

    with pytest.raises(BadInputError, match="requires finite numerical values"):
        model.fit(X, agg_client=agg_client)


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
                n_clusters=2,
                random_state=5,
            ).fit(partitions[worker_id], agg_client=agg_client)

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
    model = FederatedKMeans(**params)

    with pytest.raises(BadInputError, match=error):
        model.fit(np.array([[0.0], [1.0]]), agg_client=agg_client)


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
            n_clusters=3,
            random_state=random_state + init_idx,
            maxiter=100,
            tol=1e-8,
        ).fit(X, agg_client=agg_client)
        single_start_inertias.append(model.inertia_)

    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    multi_start = FederatedKMeans(
        n_clusters=3,
        init_method=INIT_MULTI_START_RANDOM_RANGE,
        n_init=n_init,
        random_state=random_state,
        maxiter=100,
        tol=1e-8,
    ).fit(X, agg_client=agg_client)

    assert multi_start.init_method_ == INIT_MULTI_START_RANDOM_RANGE
    assert multi_start.n_init_ == n_init
    assert multi_start.inertia_ == pytest.approx(min(single_start_inertias))
    assert multi_start.best_init_ == int(np.argmin(single_start_inertias))


def _replay_data():
    return np.array(
        [
            [0.0, 0.2],
            [0.1, -0.1],
            [-0.2, 0.0],
            [5.0, 5.2],
            [5.1, 4.9],
            [4.8, 5.0],
            [10.0, 0.0],
            [10.2, 0.2],
            [9.8, -0.1],
        ]
    )


def _agg_client():
    coordinator = AggregationCoordinator(n_workers=1)
    return SimulatedAggClient(worker_id=0, coordinator=coordinator)


def _fit_replay_model(X, *, init_method="random_range", n_init=1, random_state=17):
    return FederatedKMeans(
        n_clusters=3,
        init_method=init_method,
        n_init=n_init,
        random_state=random_state,
        tol=1e-8,
        maxiter=100,
    ).fit(X, agg_client=_agg_client(), feature_names=["x", "y"])


def test_kmeans_manual_fit_is_replayable_without_cluster_id_permutation():
    first = _fit_replay_model(_replay_data())
    second = _fit_replay_model(_replay_data())

    assert first.n_clusters == second.n_clusters
    np.testing.assert_allclose(
        first.cluster_centers_, second.cluster_centers_, atol=1e-12
    )
    np.testing.assert_array_equal(first.labels_, second.labels_)
    assert first.cluster_counts_ == second.cluster_counts_


def test_kmeans_multi_start_fit_is_replayable_with_stable_cluster_ids():
    first = _fit_replay_model(
        _replay_data(),
        init_method=INIT_MULTI_START_RANDOM_RANGE,
        n_init=4,
    )
    second = _fit_replay_model(
        _replay_data(),
        init_method=INIT_MULTI_START_RANDOM_RANGE,
        n_init=4,
    )

    np.testing.assert_allclose(
        first.cluster_centers_, second.cluster_centers_, atol=1e-12
    )
    np.testing.assert_array_equal(first.labels_, second.labels_)
    assert first.best_init_ == second.best_init_


def test_kmeans_random_state_uses_incremented_seed_for_each_start():
    model = FederatedKMeans(
        n_clusters=3,
        init_method=INIT_MULTI_START_RANDOM_RANGE,
        n_init=3,
        random_state=23,
    )
    global_min = _replay_data().min(axis=0)
    global_max = _replay_data().max(axis=0)

    for init_idx in range(3):
        expected = np.random.RandomState(23 + init_idx).uniform(
            low=global_min,
            high=global_max,
            size=(3, 2),
        )
        actual = model._initialize_centers(
            global_min=global_min,
            global_max=global_max,
            n_features=2,
            init_idx=init_idx,
        )
        np.testing.assert_allclose(actual, expected, atol=0.0)


def test_kmeans_elbow_result_can_be_replayed_manually():
    X = _replay_data()
    selection = FederatedKMeans.select_k(
        X,
        k_min=2,
        k_max=4,
        agg_client=_agg_client(),
        init_method=INIT_MULTI_START_RANDOM_RANGE,
        n_init=3,
        random_state=29,
        tol=1e-8,
        maxiter=100,
        feature_names=["x", "y"],
    )
    replay = FederatedKMeans(
        n_clusters=selection.selected_k,
        init_method=INIT_MULTI_START_RANDOM_RANGE,
        n_init=3,
        random_state=29,
        tol=1e-8,
        maxiter=100,
    ).fit(X, agg_client=_agg_client(), feature_names=["x", "y"])

    assert selection.best_model.n_clusters == replay.n_clusters
    np.testing.assert_allclose(
        selection.best_model.cluster_centers_, replay.cluster_centers_, atol=1e-12
    )
    np.testing.assert_array_equal(selection.best_model.labels_, replay.labels_)


def test_kmeans_replay_contract_preserves_feature_order():
    X = _replay_data()
    first = _fit_replay_model(X)
    reordered = _fit_replay_model(X[:, ::-1])

    first_centers = np.asarray(first.cluster_centers_)
    reordered_centers_in_first_order = np.asarray(reordered.cluster_centers_)[:, ::-1]
    cluster_id_mapping = np.asarray(
        [
            np.argmin(np.linalg.norm(first_centers - center, axis=1))
            for center in reordered_centers_in_first_order
        ]
    )

    np.testing.assert_allclose(
        first_centers[cluster_id_mapping], reordered_centers_in_first_order, atol=1e-12
    )
    np.testing.assert_array_equal(cluster_id_mapping[reordered.labels_], first.labels_)
    assert not np.array_equal(cluster_id_mapping, np.arange(first.n_clusters))
