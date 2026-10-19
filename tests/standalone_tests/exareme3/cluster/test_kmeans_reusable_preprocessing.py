import pandas as pd
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import ParseDict

from exaflow.algorithms.exareme3.cluster.kmeans import KMeansReusablePreprocessing
from exaflow.algorithms.exareme3.cluster.kmeans import (
    build_kmeans_preprocessing_parameters,
)
from exaflow.algorithms.exareme3.cluster.kmeans_cluster_creator import (
    KMeansClusterCreator,
)
from exaflow.protos.worker import worker_pb2
from exaflow.worker.exareme3.udf import udf_service


def _reusable():
    return KMeansReusablePreprocessing.model_validate(
        {
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
                "input_fingerprint": "fingerprint",
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
    )


def test_reusable_contract_stores_centers_and_no_random_replay_parameters():
    reusable = _reusable()

    assert reusable.cluster_variables == ["x", "y"]
    assert reusable.centers["cluster_2"] == {"x": 10.0, "y": 10.0}
    assert not hasattr(reusable, "base_parameters")


def test_reusable_contract_builds_full_creator_parameters():
    parameters = build_kmeans_preprocessing_parameters(
        _reusable(),
        new_column_names=["cluster"],
    )

    assert parameters == {
        "new_column_names": ["cluster"],
        "k": 2,
        "enumeration_prefix": "cluster_",
        "reusable_preprocessing": _reusable().model_dump(),
    }


def test_reusable_parameters_replay_same_assignments():
    creator = KMeansClusterCreator(
        params=build_kmeans_preprocessing_parameters(
            _reusable(), new_column_names=["cluster"]
        )
    )
    transformed = creator.transform_data(
        data=pd.DataFrame({"x": [0.0, 9.0], "y": [1.0, 10.0]})
    )

    assert transformed["cluster"].tolist() == ["cluster_1", "cluster_2"]


def test_reusable_parameters_replay_after_worker_protobuf_round_trip():
    parameters = build_kmeans_preprocessing_parameters(
        _reusable(), new_column_names=["cluster"]
    )
    request = worker_pb2.RunUdfRequest()
    ParseDict(
        {
            "preprocessing": [
                {"name": "kmeans_cluster_creator", "parameters": parameters}
            ]
        },
        request.system_args,
    )
    system_args = MessageToDict(request.system_args)
    transported_parameters = system_args["preprocessing"][0]["parameters"]
    assert type(transported_parameters["k"]) is float

    data, metadata = udf_service._apply_preprocessing_steps_to_data_and_metadata(
        data=pd.DataFrame({"x": [0.0, 9.0], "y": [1.0, 10.0]}),
        metadata={},
        preprocessing=system_args["preprocessing"],
        check_min_rows=False,
        agg_client=None,
    )
    assert data["cluster"].tolist() == ["cluster_1", "cluster_2"]
    assert metadata["cluster"]["enumerations"] == {
        "cluster_1": "cluster_1",
        "cluster_2": "cluster_2",
    }
