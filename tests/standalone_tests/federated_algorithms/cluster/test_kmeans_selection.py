import numpy as np
import pytest

from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.cluster.kmeans_selection import (
    FederatedKMeansSelector,
)
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


def test_kmeans_selector_fits_models_for_k_range_and_keeps_best_model():
    X = np.vstack(
        [
            np.tile(np.array([0.0, 0.0]), (5, 1)),
            np.tile(np.array([5.0, 5.0]), (5, 1)),
            np.tile(np.array([10.0, 10.0]), (5, 1)),
        ]
    )

    selector = FederatedKMeansSelector(
        agg_client=_agg_client(),
        k_min=2,
        k_max=4,
        random_state=4,
        maxiter=100,
        tol=1e-8,
    ).fit(X, feature_names=["age", "crp"])

    assert set(selector.models_by_k_) == {2, 3, 4}
    assert set(selector.inertia_by_k_) == {2, 3, 4}
    assert selector.selected_k_ in {2, 3, 4}
    assert isinstance(selector.best_model_, FederatedKMeans)
    assert selector.best_model_ is selector.models_by_k_[selector.selected_k_]
    assert selector.best_model_.feature_names_ == ["age", "crp"]


def test_kmeans_selector_inertia_does_not_increase_across_k_range():
    X = np.array([[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]])

    selector = FederatedKMeansSelector(
        agg_client=_agg_client(),
        k_min=1,
        k_max=3,
        random_state=5,
        maxiter=100,
    ).fit(X)

    inertias = list(selector.inertia_by_k_.values())
    assert inertias == sorted(inertias, reverse=True)


def test_kmeans_selector_rejects_invalid_k_range():
    with pytest.raises(BadInputError, match="k_min"):
        FederatedKMeansSelector(
            agg_client=_agg_client(),
            k_min=0,
            k_max=3,
        ).fit(np.array([[1.0]]))

    with pytest.raises(BadInputError, match="k_max"):
        FederatedKMeansSelector(
            agg_client=_agg_client(),
            k_min=3,
            k_max=2,
        ).fit(np.array([[1.0]]))


def test_kmeans_selector_warns_when_elbow_is_ambiguous_with_two_values():
    selector = FederatedKMeansSelector(
        agg_client=_agg_client(),
        k_min=1,
        k_max=2,
    ).fit(np.array([[0.0], [1.0], [2.0]]))

    assert selector.selected_k_ == 1
    assert selector.warning_ == "Elbow selection is ambiguous with two k values."
