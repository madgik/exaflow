from concurrent.futures import ThreadPoolExecutor
from threading import local
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.cluster.kmeans import KMeans
from exaflow.algorithms.exareme3.cluster.kmeans import _build_report
from exaflow.algorithms.exareme3.cluster.kmeans import _count_interval
from exaflow.algorithms.exareme3.cluster.kmeans import (
    build_kmeans_preprocessing_parameters,
)
from exaflow.algorithms.exareme3.cluster.kmeans import local_step
from exaflow.algorithms.exareme3.cluster.kmeans_cluster_creator import (
    KMeansClusterCreator,
)
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker import config as worker_config
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


def test_kmeans_report_rejects_small_clusters_before_generating_summaries():
    with pytest.raises(BadInputError, match="minimum cohort size"):
        _build_report(
            model=_fit_model(),
            variables=["age", "crp"],
            k_selection="manual",
            global_mean=np.array([2.55, 2.5]),
            minimum_row_count=3,
            elbow=None,
        )


def test_kmeans_report_returns_centers_at_the_minimum_cluster_size():
    report = _build_report(
        model=_fit_model(),
        variables=["age", "crp"],
        k_selection="manual",
        global_mean=np.array([2.55, 2.5]),
        minimum_row_count=2,
        elbow=None,
    )

    assert report["n_obs_interval"] == "4-5"
    assert all(cluster["center"] is not None for cluster in report["clusters"])
    assert all(cluster["profile"] for cluster in report["clusters"])
    assert all(cluster["size_interval"] == "2-3" for cluster in report["clusters"])
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
        logger=Mock(),
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


@pytest.mark.parametrize(
    "count, expected", [(0, "0"), (1, "<10"), (9, "<10"), (10, "10-19"), (20, "20-29")]
)
def test_kmeans_count_intervals_are_unchanged(count, expected):
    assert _count_interval(count, minimum_row_count=10) == expected


def test_report_rejects_small_center_support_even_with_sufficient_final_counts():
    model = _fit_model()
    model.center_support_counts_ = [1, 3]

    with pytest.raises(BadInputError, match="minimum cohort size"):
        _build_report(
            model=model,
            variables=["age", "crp"],
            k_selection="manual",
            global_mean=np.array([2.55, 2.5]),
            minimum_row_count=2,
            elbow=None,
        )


@pytest.mark.parametrize("selection", ["manual", "elbow"])
def test_privacy_failure_stops_analysis_before_replay_export(monkeypatch, selection):
    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 3)
    data = pd.DataFrame({"age": [0.0, 0.1, 5.0, 5.1], "crp": [0.0, 0.0, 5.0, 5.0]})
    client = SimulatedAggClient(worker_id=0, coordinator=AggregationCoordinator(1))
    algorithm = KMeans(
        engine=SimpleNamespace(),
        logger=Mock(),
        inputdata=Inputdata(
            data_model="dm:0.1", datasets=["dataset"], variables=["age", "crp"]
        ),
        y=["age", "crp"],
        parameters={"k_selection": selection, "k": 2, "k_min": 2, "k_max": 2},
    )
    monkeypatch.setattr(
        algorithm,
        "run_local_udf",
        lambda **kwargs: local_step(data=data, agg_client=client, **kwargs["kw_args"]),
    )
    export = Mock()
    monkeypatch.setattr(
        "exaflow.algorithms.exareme3.cluster.kmeans._build_reusable_preprocessing",
        export,
    )
    with pytest.raises(BadInputError, match="minimum (cohort|cluster) size"):
        algorithm.run()
    export.assert_not_called()


@pytest.mark.parametrize("n_workers", [1, 5])
@pytest.mark.parametrize("selection", ["manual", "elbow"])
def test_report_privacy_uses_global_counts_and_workers_agree(
    monkeypatch, n_workers, selection
):
    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 3)
    data = pd.DataFrame({"x": np.repeat([0.0, 10.0], 10)})
    coordinator = AggregationCoordinator(n_workers)

    def run(worker_id):
        try:
            return local_step(
                agg_client=SimulatedAggClient(
                    worker_id=worker_id, coordinator=coordinator
                ),
                data=data.iloc[worker_id::n_workers],
                variables=["x"],
                k_selection=selection,
                n_clusters=2,
                k_min=1,
                k_max=2,
                tol=1e-4,
                maxiter=10,
                init_method="multi_start_random_range",
                n_init=3,
            )
        except Exception as exc:
            coordinator.abort(exc)
            raise

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run, worker_id) for worker_id in range(n_workers)]
        reports = [future.result(timeout=10) for future in futures]

    assert all(report == reports[0] for report in reports)
    assert all(cluster["center"] for cluster in reports[0]["clusters"])


@pytest.mark.parametrize("n_workers", [1, 5])
def test_report_uses_strictest_participating_worker_floor(monkeypatch, n_workers):
    settings = local()

    class PrivacyConfig:
        @property
        def minimum_row_count(self):
            return settings.minimum

    monkeypatch.setattr(worker_config, "privacy", PrivacyConfig())
    data = pd.DataFrame({"x": np.repeat([0.0, 10.0], 10)})
    coordinator = AggregationCoordinator(n_workers)

    def run(worker_id):
        settings.minimum = 11 if worker_id == 0 else 1
        try:
            local_step(
                agg_client=SimulatedAggClient(
                    worker_id=worker_id, coordinator=coordinator
                ),
                data=data.iloc[worker_id::n_workers],
                variables=["x"],
                k_selection="manual",
                n_clusters=2,
                k_min=1,
                k_max=2,
                tol=1e-4,
                maxiter=10,
            )
        except BadInputError as exc:
            return str(exc)
        except Exception as exc:
            coordinator.abort(exc)
            raise
        return None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run, worker_id) for worker_id in range(n_workers)]
        errors = [future.result(timeout=10) for future in futures]

    assert all(error == errors[0] for error in errors)
    assert "minimum cohort size" in errors[0]


@pytest.mark.parametrize("selection", ["manual", "elbow"])
def test_fitted_count_and_custom_prefix_survive_replay(monkeypatch, selection):
    report = _build_report(
        model=_fit_model(),
        variables=["age", "crp"],
        k_selection=selection,
        global_mean=np.array([2.55, 2.5]),
        minimum_row_count=1,
        enumeration_prefix="group__9_",
        elbow=None
        if selection == "manual"
        else {
            "k_min": 1,
            "k_max": 4,
            "selected_k": 2,
            "inertia_by_k": {2: 0.1},
        },
    )
    algorithm = KMeans(
        engine=SimpleNamespace(),
        logger=Mock(),
        inputdata=Inputdata(
            data_model="dm:0.1", datasets=["dataset"], variables=["age", "crp"]
        ),
        y=["age", "crp"],
        parameters={
            "k_selection": selection,
            "k": 4,
            "enumeration_prefix": "group__9_",
        },
    )
    calls = []

    def fitted_report(**kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(algorithm, "run_local_udf", fitted_report)

    result = algorithm.run()
    parameters = build_kmeans_preprocessing_parameters(
        result.reusable_preprocessing,
        new_column_names=["group"],
    )
    assert calls[0]["kw_args"]["enumeration_prefix"] == "group__9_"
    assert parameters["k"] == result.selected_k == 2
    assert parameters["enumeration_prefix"] == "group__9_"
    assert [cluster.cluster_id for cluster in result.clusters] == [
        "group__9_1",
        "group__9_2",
    ]
    transformed = KMeansClusterCreator(params=parameters).transform_data(
        data=pd.DataFrame({"age": [0.0, 5.0], "crp": [0.0, 5.0]}),
    )
    assert set(transformed["group"]) == {"group__9_1", "group__9_2"}


def test_custom_prefix_and_one_based_ids_in_empty_cluster_warnings():
    model = _fit_model()
    model.empty_clusters_ = [0]
    report = _build_report(
        model=model,
        variables=["age", "crp"],
        k_selection="manual",
        global_mean=np.array([2.55, 2.5]),
        minimum_row_count=1,
        elbow=None,
        enumeration_prefix="segment_",
    )
    assert "Empty clusters were produced: segment_1." in report["warnings"]
    assert report["clusters"][0]["label"] == "Cluster 1"
    assert "segment_1" in report["clusters"][0]["interpretation"]
