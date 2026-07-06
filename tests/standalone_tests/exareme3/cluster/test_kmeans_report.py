import numpy as np

from exaflow.algorithms.exareme3.cluster.kmeans import K_SELECTION_ELBOW
from exaflow.algorithms.exareme3.cluster.kmeans import K_SELECTION_MANUAL
from exaflow.algorithms.exareme3.cluster.kmeans import _build_report
from exaflow.algorithms.exareme3.cluster.kmeans import _compactness_labels
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _fit_model(X, *, n_clusters=2):
    coordinator = AggregationCoordinator(n_workers=1)
    agg_client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    return FederatedKMeans(
        agg_client=agg_client,
        n_clusters=n_clusters,
        random_state=3,
        maxiter=100,
        tol=1e-8,
    ).fit(X, feature_names=["age", "crp"])


def test_kmeans_report_returns_named_centers_and_count_intervals():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ]
    )
    model = _fit_model(X)

    report = _build_report(
        model=model,
        variables=["age", "crp"],
        k_selection=K_SELECTION_MANUAL,
        global_mean=np.mean(X, axis=0),
        minimum_row_count=2,
        elbow=None,
    )

    assert report["title"] == "K-Means Cluster Report"
    assert report["result_type"] == "privacy_safe_cluster_report"
    assert report["variables"] == ["age", "crp"]
    assert report["k_selection"] == K_SELECTION_MANUAL
    assert report["selected_k"] == 2
    assert report["initialization_method"] == "random_range"
    assert report["n_init"] == 1
    assert report["selected_initialization"] == 0
    assert report["n_obs_interval"] == "4-5"
    assert "not a real patient" in report["center_definition"]
    assert "Explore baseline covariate patterns." in report["intended_use"]
    assert "intervals" in report["privacy_note"]
    assert any("not diagnoses" in item for item in report["limitations"])
    assert len(report["clusters"]) == 2
    assert all(cluster["center"] is not None for cluster in report["clusters"])
    assert set(report["clusters"][0]["center"]) == {"age", "crp"}
    assert report["clusters"][0]["label"].startswith("Cluster ")
    assert report["clusters"][0]["interpretation"]
    assert report["warnings"] == []


def test_kmeans_report_suppresses_small_cluster_centers():
    X = np.array(
        [
            [0.0, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ]
    )
    model = _fit_model(X)

    report = _build_report(
        model=model,
        variables=["age", "crp"],
        k_selection=K_SELECTION_MANUAL,
        global_mean=np.mean(X, axis=0),
        minimum_row_count=2,
        elbow=None,
    )

    suppressed = [
        cluster for cluster in report["clusters"] if cluster["center"] is None
    ]
    assert suppressed
    assert "privacy threshold" in suppressed[0]["interpretation"]
    assert any("suppressed by privacy threshold" in msg for msg in report["warnings"])


def test_kmeans_report_includes_elbow_payload_without_models():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ]
    )
    model = _fit_model(X)

    report = _build_report(
        model=model,
        variables=["age", "crp"],
        k_selection=K_SELECTION_ELBOW,
        global_mean=np.mean(X, axis=0),
        minimum_row_count=2,
        elbow={
            "k_min": 2,
            "k_max": 3,
            "selected_k": 2,
            "inertia_by_k": {2: 10.0, 3: 5.0},
            "warning": "Elbow selection is ambiguous with two k values.",
        },
    )

    assert report["elbow"] == {
        "k_min": 2,
        "k_max": 3,
        "selected_k": 2,
        "inertia_by_k": {2: 10.0, 3: 5.0},
        "warning": "Elbow selection is ambiguous with two k values.",
    }
    assert "models_by_k" not in report["elbow"]


def test_compactness_labels_exclude_suppressed_clusters():
    labels = _compactness_labels(
        counts=[1, 10, 10],
        cluster_inertia=[0.1, 1.0, 100.0],
        can_show=[False, True, True],
    )

    assert set(labels) == {1, 2}
    assert labels[1] == "high"
    assert labels[2] == "low"
