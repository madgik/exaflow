import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.cluster.kmeans import make_kmeans_input_fingerprint
from exaflow.algorithms.exareme3.preprocessing.kmeans_cluster_creator import (
    KMeansClusterCreator,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput


def _params(**replay_overrides):
    replay_payload = {
        "schema_version": "1",
        "preprocessing_name": "kmeans_cluster_creator",
        "cluster_variables": ["x", "y"],
        "centers": {
            "cluster_0": {"x": 0.0, "y": 0.0},
            "cluster_1": {"x": 10.0, "y": 10.0},
        },
        "source_context": {
            "data_model": "dm:0.1",
            "datasets": ["dataset"],
            "input_fingerprint": make_kmeans_input_fingerprint(
                data_model="dm:0.1",
                datasets=["dataset"],
                filters=None,
                feature_names=["x", "y"],
            ),
        },
        "available_outputs": [
            {
                "output_mode": "full",
                "semantic_operation": "all_clusters",
                "variable_type": "nominal",
                "number_of_variables": 1,
                "eligible_roles": ["predictor"],
                "cardinality": 2,
                "selection_rule": "all clusters",
            }
        ],
        "cluster_choices": [
            {"cluster_id": "cluster_0", "label": "Cluster 0"},
            {"cluster_id": "cluster_1", "label": "Cluster 1"},
        ],
    }
    replay_payload.update(replay_overrides)
    return {
        "code": "cluster",
        "reusable_preprocessing": replay_payload,
    }


def test_specification_exposes_only_full_replay_inputs():
    spec = KMeansClusterCreator.get_specification()

    assert spec.label == "KMeans Column Creator"
    assert set(spec.parameters) == {"code", "reusable_preprocessing"}
    assert spec.components == []


def test_creator_assigns_rows_to_supplied_centers_without_aggregation():
    creator = KMeansClusterCreator(params=_params())
    data = pd.DataFrame({"x": [0.1, 9.9, 4.0], "y": [0.0, 10.1, 4.0]})

    transformed = creator.transform_data(data=data)

    assert transformed["cluster"].tolist() == ["cluster_0", "cluster_1", "cluster_0"]


@pytest.mark.parametrize(
    ("inputdata", "message"),
    [
        (
            Inputdata(
                data_model="other:0.1",
                datasets=["dataset"],
                filters=None,
                variables=["x", "y"],
            ),
            "data model",
        ),
        (
            Inputdata(
                data_model="dm:0.1",
                datasets=["other_dataset"],
                filters=None,
                variables=["x", "y"],
            ),
            "datasets",
        ),
        (
            Inputdata(
                data_model="dm:0.1",
                datasets=["dataset"],
                filters={"condition": "AND", "rules": []},
                variables=["x", "y"],
            ),
            "filters",
        ),
    ],
)
def test_creator_rejects_replay_from_different_input(inputdata, message):
    creator = KMeansClusterCreator(params=_params())

    with pytest.raises(BadUserInput, match=message):
        creator.validate_params(inputdata=inputdata, metadata={})


def test_creator_accepts_replay_from_matching_input():
    creator = KMeansClusterCreator(params=_params())
    inputdata = Inputdata(
        data_model="dm:0.1",
        datasets=["dataset"],
        filters=None,
        variables=["x", "y"],
    )

    creator.validate_params(inputdata=inputdata, metadata={})


def test_creator_rejects_invalid_replay_fingerprint():
    creator = KMeansClusterCreator(
        params=_params(
            source_context={
                "data_model": "dm:0.1",
                "datasets": ["dataset"],
                "input_fingerprint": "invalid",
            }
        )
    )
    inputdata = Inputdata(
        data_model="dm:0.1",
        datasets=["dataset"],
        filters=None,
        variables=["x", "y"],
    )

    with pytest.raises(BadUserInput, match="fingerprint"):
        creator.validate_params(inputdata=inputdata, metadata={})


def test_creator_rejects_replay_when_clustering_features_are_not_requested():
    creator = KMeansClusterCreator(params=_params())
    inputdata = Inputdata(
        data_model="dm:0.1",
        datasets=["dataset"],
        filters=None,
        variables=["outcome"],
    )

    with pytest.raises(BadUserInput, match="missing from the current input"):
        creator.validate_params(inputdata=inputdata, metadata={})


def test_creator_rejects_center_missing_a_clustering_feature():
    params = _params()
    del params["reusable_preprocessing"]["centers"]["cluster_1"]["y"]

    with pytest.raises(BadUserInput, match="Center 'cluster_1'.*features:.*'y'"):
        KMeansClusterCreator(params=params)


def test_creator_metadata_contains_all_fitted_clusters():
    creator = KMeansClusterCreator(params=_params())

    metadata = creator.transform_metadata(metadata={})

    assert metadata["cluster"]["is_categorical"]
    assert metadata["cluster"]["enumerations"] == {
        "cluster_0": "cluster_0",
        "cluster_1": "cluster_1",
    }


def test_creator_rejects_non_finite_data():
    creator = KMeansClusterCreator(params=_params())

    with pytest.raises(ValueError, match="finite numerical values"):
        creator.transform_data(data=pd.DataFrame({"x": [np.nan], "y": [0.0]}))
