import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.cluster.kmeans import make_kmeans_input_fingerprint
from exaflow.algorithms.exareme3.cluster.kmeans_cluster_creator import (
    KMeansClusterCreator,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput


def _params(**replay_overrides):
    replay_payload = {
        "enumeration_prefix": "cluster_",
        "preprocessing_name": "kmeans_cluster_creator",
        "cluster_variables": ["x", "y"],
        "centers": {
            "cluster_1": {"x": 0.0, "y": 0.0},
            "cluster_2": {"x": 10.0, "y": 10.0},
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
            {"cluster_id": "cluster_1", "label": "Cluster 1"},
            {"cluster_id": "cluster_2", "label": "Cluster 2"},
        ],
    }
    replay_payload.update(replay_overrides)
    return {
        "new_column_names": ["cluster"],
        "k": 2,
        "enumeration_prefix": "cluster_",
        "reusable_preprocessing": replay_payload,
    }


def test_specification_exposes_only_full_replay_inputs():
    spec = KMeansClusterCreator.get_specification()

    assert spec.label == "KMeans Column Creator"
    assert set(spec.parameters) == {
        "new_column_names",
        "k",
        "enumeration_prefix",
        "reusable_preprocessing",
    }
    assert spec.components == []


def test_creator_assigns_rows_to_supplied_centers_without_aggregation():
    creator = KMeansClusterCreator(params=_params())
    data = pd.DataFrame({"x": [0.1, 9.9, 4.0], "y": [0.0, 10.1, 4.0]})

    transformed = creator.transform_data(data=data)

    assert transformed["cluster"].tolist() == ["cluster_1", "cluster_2", "cluster_1"]


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
    del params["reusable_preprocessing"]["centers"]["cluster_2"]["y"]

    with pytest.raises(BadUserInput, match="Center 'cluster_2'.*features:.*'y'"):
        KMeansClusterCreator(params=params)


def test_creator_metadata_contains_all_fitted_clusters():
    creator = KMeansClusterCreator(params=_params())

    metadata = creator.transform_metadata(metadata={})

    assert metadata["cluster"]["is_categorical"]
    assert metadata["cluster"]["enumerations"] == {
        "cluster_1": "cluster_1",
        "cluster_2": "cluster_2",
    }


def test_creator_rejects_non_finite_data():
    creator = KMeansClusterCreator(params=_params())

    with pytest.raises(ValueError, match="finite numerical values"):
        creator.transform_data(data=pd.DataFrame({"x": [np.nan], "y": [0.0]}))


@pytest.mark.parametrize(
    "key,value", [("k", 1), ("k", 3), ("enumeration_prefix", "other_")]
)
def test_creator_rejects_summary_that_disagrees_with_replay(key, value):
    params = _params()
    params[key] = value
    with pytest.raises(BadUserInput, match="count or prefix"):
        KMeansClusterCreator(params=params)


def test_creator_uses_numeric_order_for_custom_prefix_and_twelve_clusters():
    prefix = "group__9_"
    ids = [f"{prefix}{i}" for i in range(1, 13)]
    params = _params(
        enumeration_prefix=prefix,
        centers={
            name: {"x": float(i), "y": float(i)}
            for i, name in reversed(list(enumerate(ids, 1)))
        },
        cluster_choices=[{"cluster_id": name, "label": name} for name in ids],
    )
    params.update(k=12, enumeration_prefix=prefix)
    creator = KMeansClusterCreator(params=params)
    transformed, metadata = creator.transform_data_and_metadata(
        data=pd.DataFrame({"x": [1.0, 2.0, 10.0, 12.0], "y": [1.0, 2.0, 10.0, 12.0]}),
        metadata={},
    )
    assert transformed["cluster"].tolist() == [ids[0], ids[1], ids[9], ids[11]]
    assert list(metadata["cluster"]["enumerations"]) == ids


@pytest.mark.parametrize(
    "override",
    [
        {"centers": {}},
        {"enumeration_prefix": "other_"},
        {"cluster_choices": []},
    ],
)
def test_creator_rejects_invalid_replay_schema(override):
    with pytest.raises(BadUserInput, match="replay payload"):
        KMeansClusterCreator(params=_params(**override))
