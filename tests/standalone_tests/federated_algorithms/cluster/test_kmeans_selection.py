import numpy as np
import pytest

from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeansResults
from exaflow.algorithms.federated.cluster.kmeans import NoEligibleKMeansModelError
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _agg_client():
    coordinator = AggregationCoordinator(n_workers=1)
    return SimulatedAggClient(worker_id=0, coordinator=coordinator)


def test_kmeans_select_k_fits_models_for_k_range_and_keeps_best_model():
    X = np.vstack(
        [
            np.tile(np.array([0.0, 0.0]), (5, 1)),
            np.tile(np.array([5.0, 5.0]), (5, 1)),
            np.tile(np.array([10.0, 10.0]), (5, 1)),
        ]
    )

    selection = FederatedKMeans.select_k(
        X,
        k_min=2,
        k_max=4,
        agg_client=_agg_client(),
        random_state=4,
        maxiter=100,
        tol=1e-8,
        feature_names=["age", "crp"],
    )

    assert set(selection.models_by_k) == {2, 3, 4}
    assert set(selection.inertia_by_k) == {2, 3, 4}
    assert selection.selected_k in {2, 3, 4}
    assert isinstance(selection.best_model, FederatedKMeansResults)
    assert selection.best_model is selection.models_by_k[selection.selected_k]
    assert selection.best_model.feature_names_ == ["age", "crp"]


def test_kmeans_select_k_inertia_does_not_increase_across_k_range():
    X = np.array([[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]])

    selection = FederatedKMeans.select_k(
        X,
        k_min=1,
        k_max=3,
        agg_client=_agg_client(),
        random_state=5,
        maxiter=100,
    )

    inertias = list(selection.inertia_by_k.values())
    assert inertias == sorted(inertias, reverse=True)


def test_kmeans_select_k_rejects_invalid_k_range():
    with pytest.raises(BadInputError, match="k_min"):
        FederatedKMeans.select_k(
            np.array([[1.0]]),
            k_min=0,
            k_max=3,
            agg_client=_agg_client(),
        )

    with pytest.raises(BadInputError, match="k_max"):
        FederatedKMeans.select_k(
            np.array([[1.0]]),
            k_min=3,
            k_max=2,
            agg_client=_agg_client(),
        )


def test_kmeans_select_k_warns_when_elbow_is_ambiguous_with_two_values():
    selection = FederatedKMeans.select_k(
        np.array([[0.0], [1.0], [2.0]]),
        k_min=1,
        k_max=2,
        agg_client=_agg_client(),
    )

    assert selection.selected_k == 1
    assert selection.warning == "Elbow selection is ambiguous with two k values."


@pytest.fixture
def three_initializations(monkeypatch):
    # Controlled fits isolate the selection policy from numerical convergence.
    candidates = []
    for init_idx, (counts, inertia) in enumerate(
        [([79, 20, 1], 100.0), ([60, 25, 15], 120.0), ([50, 30, 20], 130.0)]
    ):
        candidates.append(
            {
                "centers": np.array([[0.0], [5.0], [10.0]]),
                "labels": np.repeat(np.arange(3), counts),
                "counts": np.asarray(counts),
                "center_support_counts": np.asarray(counts),
                "cluster_inertia": np.array([inertia, 0.0, 0.0]),
                "inertia": inertia,
                "n_iter": 1,
                "converged": True,
                "init_idx": init_idx,
            }
        )
    monkeypatch.setattr(
        FederatedKMeans,
        "_fit_one_initialization",
        lambda self, **kwargs: candidates[kwargs["init_idx"]],
    )
    return candidates


def test_multi_start_keeps_lowest_inertia_eligible_fit(three_initializations):
    result = FederatedKMeans(
        n_clusters=3, init_method="multi_start_random_range", n_init=3
    ).fit(
        np.arange(100).reshape(-1, 1), agg_client=_agg_client(), minimum_cluster_size=10
    )

    assert result.best_init_ == 1
    assert result.inertia_ == 120.0
    assert result.cluster_counts_ == [60, 25, 15]


def test_unfiltered_multi_start_preserves_manual_selection(three_initializations):
    result = FederatedKMeans(
        n_clusters=3, init_method="multi_start_random_range", n_init=3
    ).fit(np.arange(100).reshape(-1, 1), agg_client=_agg_client())

    assert result.best_init_ == 0
    assert result.inertia_ == 100.0
    assert not result.meets_minimum_cluster_size(10)


def test_multi_start_rejects_when_all_initializations_are_ineligible(
    three_initializations,
):
    with pytest.raises(NoEligibleKMeansModelError, match="minimum cluster size"):
        FederatedKMeans(
            n_clusters=3, init_method="multi_start_random_range", n_init=3
        ).fit(
            np.arange(100).reshape(-1, 1),
            agg_client=_agg_client(),
            minimum_cluster_size=31,
        )


def test_privacy_uses_centroid_contributors_after_early_stopping(monkeypatch):
    monkeypatch.setattr(
        FederatedKMeans,
        "_initialize_centers",
        lambda self, **kwargs: np.array([[0.0], [0.1]]),
    )
    X = np.arange(5, dtype=float).reshape(-1, 1)
    model = FederatedKMeans(n_clusters=2, maxiter=1, tol=0)
    result = model.fit(X, agg_client=_agg_client())

    assert result.cluster_counts_ == [2, 3]
    assert result.center_support_counts_ == [1, 4]
    assert not result.meets_minimum_cluster_size(2)
    assert result.n_iter_ == 1

    with pytest.raises(NoEligibleKMeansModelError):
        model.fit(X, agg_client=_agg_client(), minimum_cluster_size=2)


@pytest.mark.parametrize("threshold", [0, -1, 1.5, True])
def test_privacy_threshold_must_be_a_positive_integer(threshold):
    with pytest.raises(BadInputError, match="minimum_cluster_size"):
        FederatedKMeans(n_clusters=1).fit(
            np.array([[0.0], [1.0]]),
            agg_client=_agg_client(),
            minimum_cluster_size=threshold,
        )


def test_privacy_filter_rejects_empty_input():
    with pytest.raises(NoEligibleKMeansModelError):
        FederatedKMeans(n_clusters=1).fit(
            np.empty((0, 1)), agg_client=_agg_client(), minimum_cluster_size=2
        )


def test_elbow_omits_ineligible_models_and_their_inertias():
    X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
    selection = FederatedKMeans.select_k(
        X, k_min=1, k_max=2, agg_client=_agg_client(), minimum_cluster_size=4
    )

    assert selection.selected_k == 1
    assert set(selection.models_by_k) == {1}
    assert set(selection.inertia_by_k) == {1}
    assert "excluded" in selection.warning
    assert selection.best_model.meets_minimum_cluster_size(4)


def test_elbow_fails_when_no_k_is_eligible():
    with pytest.raises(NoEligibleKMeansModelError, match="requested k range"):
        FederatedKMeans.select_k(
            np.array([[0.0], [0.1], [10.0], [10.1]]),
            k_min=2,
            k_max=3,
            agg_client=_agg_client(),
            minimum_cluster_size=5,
        )


def test_elbow_does_not_swallow_unrelated_fit_errors():
    with pytest.raises(BadInputError, match="maxiter"):
        FederatedKMeans.select_k(
            np.array([[0.0], [1.0]]),
            k_min=1,
            k_max=2,
            agg_client=_agg_client(),
            minimum_cluster_size=2,
            maxiter=0,
        )
