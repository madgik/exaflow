import pandas as pd

from exaflow.algorithms.exareme3.cluster.kmeans import KMeansReusablePreprocessing
from exaflow.algorithms.exareme3.cluster.kmeans import (
    build_kmeans_preprocessing_parameters,
)
from exaflow.algorithms.exareme3.preprocessing.kmeans_cluster_creator import (
    KMeansClusterCreator,
)


def _reusable():
    return KMeansReusablePreprocessing.model_validate(
        {
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
                {"cluster_id": "cluster_0", "label": "Cluster 0"},
                {"cluster_id": "cluster_1", "label": "Cluster 1"},
            ],
        }
    )


def test_reusable_contract_stores_centers_and_no_random_replay_parameters():
    reusable = _reusable()

    assert reusable.cluster_variables == ["x", "y"]
    assert reusable.centers["cluster_1"] == {"x": 10.0, "y": 10.0}
    assert not hasattr(reusable, "base_parameters")


def test_reusable_contract_builds_full_creator_parameters():
    parameters = build_kmeans_preprocessing_parameters(
        _reusable(),
        code="cluster",
    )

    assert parameters == {
        "code": "cluster",
        "reusable_preprocessing": _reusable().model_dump(),
    }


def test_reusable_parameters_replay_same_assignments():
    creator = KMeansClusterCreator(
        params=build_kmeans_preprocessing_parameters(_reusable(), code="cluster")
    )
    transformed = creator.transform_data(
        data=pd.DataFrame({"x": [0.0, 9.0], "y": [1.0, 10.0]})
    )

    assert transformed["cluster"].tolist() == ["cluster_0", "cluster_1"]
