from types import SimpleNamespace

import numpy as np

from exaflow.algorithms.exareme3.cluster.kmeans import KMeans
from exaflow.algorithms.exareme3.cluster.kmeans import _build_report
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _fit_model():
    data = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    client = SimulatedAggClient(
        worker_id=0, coordinator=AggregationCoordinator(n_workers=1)
    )
    return FederatedKMeans(n_clusters=2, random_state=3, tol=1e-8).fit(
        data, agg_client=client, feature_names=["age", "crp"]
    )


def test_kmeans_report_always_returns_centers_and_masks_only_counts():
    report = _build_report(
        model=_fit_model(),
        variables=["age", "crp"],
        k_selection="manual",
        global_mean=np.array([2.55, 2.5]),
        minimum_row_count=3,
        elbow=None,
    )

    assert report["n_obs_interval"] == "3-5"
    assert all(cluster["center"] is not None for cluster in report["clusters"])
    assert all(cluster["profile"] for cluster in report["clusters"])
    assert all(cluster["size_interval"] == "<3" for cluster in report["clusters"])
    assert not any("suppressed" in warning for warning in report["warnings"])


def test_kmeans_run_stores_centers_in_reusable_contract(monkeypatch):
    model = _fit_model()
    report = _build_report(
        model=model,
        variables=["age", "crp"],
        k_selection="manual",
        global_mean=np.array([2.55, 2.5]),
        minimum_row_count=1,
        elbow=None,
    )
    algorithm = KMeans(
        engine=SimpleNamespace(),
        inputdata=Inputdata(
            data_model="dm:0.1",
            datasets=["dataset"],
            variables=["age", "crp"],
            filters=None,
        ),
        y=["age", "crp"],
        parameters={},
    )
    monkeypatch.setattr(algorithm, "run_local_udf", lambda **kwargs: report)

    result = algorithm.run()

    assert result.reusable_preprocessing.cluster_variables == ["age", "crp"]
    assert result.reusable_preprocessing.centers == {
        cluster["cluster_id"]: cluster["center"] for cluster in report["clusters"]
    }
    assert [
        output.output_mode for output in result.reusable_preprocessing.available_outputs
    ] == ["full"]


def test_kmeans_specification_still_exposes_fitting_seed_only_for_fitting():
    random_state = KMeans.get_specification().parameters["random_state"]

    assert random_state.default == 123
    assert random_state.min == 0
    assert random_state.max == 4294967276


def test_kmeans_report_preserves_all_count_interval_categories():
    model = _fit_model()
    model.cluster_counts_ = [0, 12]
    model.n_obs_ = 12

    report = _build_report(
        model=model,
        variables=["age", "crp"],
        k_selection="manual",
        global_mean=np.array([2.55, 2.5]),
        minimum_row_count=10,
        elbow=None,
    )

    assert [cluster["size_interval"] for cluster in report["clusters"]] == [
        "0",
        "10-19",
    ]
    assert all(cluster["center"] for cluster in report["clusters"])
