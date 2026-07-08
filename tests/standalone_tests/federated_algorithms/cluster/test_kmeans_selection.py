import numpy as np
import pytest

from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeansResults
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
