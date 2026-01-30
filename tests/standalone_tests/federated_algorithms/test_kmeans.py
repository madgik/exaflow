import numpy as np

from exaflow.algorithms.federated.kmeans import FederatedKMeans
from tests.standalone_tests.federated_algorithms.utils import DummyAggClient


class TrackingDummyAggClient(DummyAggClient):
    def __init__(self):
        super().__init__()
        self.sum_calls = 0

    def sum(self, value):
        self.sum_calls += 1
        return super().sum(value)


def _make_blob_centers():
    """Return two tight clusters around (0, 0) and (5, 0)."""
    cluster_a = np.tile(np.array([0.0, 0.0]), (5, 1))
    cluster_b = np.tile(np.array([5.0, 0.0]), (5, 1))
    return np.vstack([cluster_a, cluster_b])


def test_fit_returns_total_observations():
    agg_client = DummyAggClient()
    model = FederatedKMeans(agg_client=agg_client, n_clusters=2, random_state=0)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    model.fit(X)
    assert model.n_obs_ == X.shape[0]


def test_fit_centers_shape_matches_k():
    agg_client = DummyAggClient()
    model = FederatedKMeans(agg_client=agg_client, n_clusters=3, random_state=1)
    X = np.random.RandomState(1).randn(15, 2)
    model.fit(X)
    centers = np.asarray(model.cluster_centers_)
    assert centers.shape == (3, 2)


def test_fit_single_cluster_center_near_mean():
    agg_client = DummyAggClient()
    model = FederatedKMeans(agg_client=agg_client, n_clusters=1, random_state=2)
    X = np.array([[1.0, 2.0], [1.2, 1.8], [0.9, 2.1]])
    model.fit(X)
    center = np.asarray(model.cluster_centers_)[0]
    assert np.allclose(center, np.mean(X, axis=0), atol=1e-6)


def test_fit_two_clusters_find_expected_means():
    agg_client = DummyAggClient()
    model = FederatedKMeans(agg_client=agg_client, n_clusters=2, random_state=3)
    X = _make_blob_centers()
    model.fit(X)
    centers = np.asarray(model.cluster_centers_)
    assert any(
        np.linalg.norm(center - np.array([0.0, 0.0])) < 1e-6 for center in centers
    )
    assert any(
        np.linalg.norm(center - np.array([5.0, 0.0])) < 1e-6 for center in centers
    )


def test_fit_returns_empty_if_no_samples():
    agg_client = DummyAggClient()
    model = FederatedKMeans(agg_client=agg_client, n_clusters=2, random_state=4)
    X = np.zeros((0, 3))
    model.fit(X)
    assert model.n_obs_ == 0
    assert model.cluster_centers_ == []


def test_fit_single_feature_data():
    agg_client = DummyAggClient()
    model = FederatedKMeans(agg_client=agg_client, n_clusters=2, random_state=5)
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    model.fit(X)
    centers = np.asarray(model.cluster_centers_)
    assert centers.shape == (2, 1)


def test_fit_handles_empty_cluster_resets_to_zero():
    agg_client = DummyAggClient()
    model = FederatedKMeans(agg_client=agg_client, n_clusters=4, random_state=6)
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    model.fit(X)
    centers = np.asarray(model.cluster_centers_)
    assert any(np.allclose(center, 0.0, atol=1e-8) for center in centers)


def test_fit_random_state_is_reproducible():
    agg_client = DummyAggClient()
    model_a = FederatedKMeans(agg_client=agg_client, n_clusters=2, random_state=7)
    model_b = FederatedKMeans(agg_client=agg_client, n_clusters=2, random_state=7)
    X = _make_blob_centers()
    centers_a = np.asarray(model_a.fit(X).cluster_centers_)
    centers_b = np.asarray(model_b.fit(X).cluster_centers_)
    assert np.allclose(centers_a, centers_b)


def test_fit_supports_high_dim_data():
    agg_client = DummyAggClient()
    model = FederatedKMeans(agg_client=agg_client, n_clusters=3, random_state=8)
    X = np.random.RandomState(8).randn(20, 8)
    model.fit(X)
    centers = np.asarray(model.cluster_centers_)
    assert centers.shape == (3, 8)
    assert model.n_obs_ == 20


def test_fit_calls_aggregator_sum_multiple_times():
    agg_client = TrackingDummyAggClient()
    model = FederatedKMeans(agg_client=agg_client, n_clusters=2, random_state=9)
    X = _make_blob_centers()
    model.fit(X)
    assert agg_client.sum_calls >= 3
