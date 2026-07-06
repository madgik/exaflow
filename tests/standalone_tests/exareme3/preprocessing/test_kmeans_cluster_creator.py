from types import SimpleNamespace

import pandas as pd
import pytest

from exaflow.algorithms.exareme3.cluster.kmeans import K_SELECTION_ELBOW
from exaflow.algorithms.exareme3.cluster.kmeans import K_SELECTION_MANUAL
from exaflow.algorithms.exareme3.preprocessing.kmeans_cluster_creator import (
    OUTPUT_MODE_BINARY,
)
from exaflow.algorithms.exareme3.preprocessing.kmeans_cluster_creator import (
    OUTPUT_MODE_FULL,
)
from exaflow.algorithms.exareme3.preprocessing.kmeans_cluster_creator import (
    OUTPUT_MODE_SUBSET,
)
from exaflow.algorithms.exareme3.preprocessing.kmeans_cluster_creator import (
    KMeansClusterCreator,
)
from exaflow.algorithms.federated.cluster.kmeans import INIT_MULTI_START_RANDOM_RANGE
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _agg_client():
    coordinator = AggregationCoordinator(n_workers=1)
    return SimulatedAggClient(worker_id=0, coordinator=coordinator)


def _inputdata():
    return Inputdata(
        data_model="dm:0.1",
        datasets=["d1"],
        variables=["age", "crp", "outcome"],
    )


def _metadata():
    return {
        "age": {"is_categorical": False, "sql_type": "real"},
        "crp": {"is_categorical": False, "sql_type": "real"},
        "outcome": {"is_categorical": False, "sql_type": "real"},
    }


def _data_three_clusters():
    return pd.DataFrame(
        {
            "age": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
            "crp": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2],
            "outcome": [1.0] * 9,
        },
        index=[10, 11, 12, 20, 21, 22, 30, 31, 32],
    )


def _creator(params):
    return KMeansClusterCreator(params=params)


def test_get_specification_exposes_new_categorical_output():
    spec = KMeansClusterCreator.get_specification()

    assert spec.name == "kmeans_cluster_creator"
    assert spec.output.code_parameter == "code"
    assert spec.components[0].value == "AGGREGATION_SERVER"
    assert "init_method" in spec.parameters
    assert "n_init" in spec.parameters


def test_transform_metadata_for_full_manual_mode():
    creator = _creator(
        {
            "code": "kmeans_cluster",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_FULL,
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    transformed = creator.transform_metadata(metadata=_metadata())

    assert transformed["kmeans_cluster"]["is_categorical"] is True
    assert transformed["kmeans_cluster"]["enumerations"] == {
        "cluster_0": "cluster_0",
        "cluster_1": "cluster_1",
        "cluster_2": "cluster_2",
    }


def test_transform_metadata_for_full_elbow_mode_uses_k_max_as_upper_bound():
    creator = _creator(
        {
            "code": "kmeans_cluster",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_ELBOW,
            "k_min": 2,
            "k_max": 4,
            "output_mode": OUTPUT_MODE_FULL,
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    transformed = creator.transform_metadata(metadata=_metadata())

    assert set(transformed["kmeans_cluster"]["enumerations"]) == {
        "cluster_0",
        "cluster_1",
        "cluster_2",
        "cluster_3",
    }


def test_runtime_metadata_for_full_elbow_mode_uses_selected_k(monkeypatch):
    from exaflow.worker import config as worker_config

    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 1)
    creator = _creator(
        {
            "code": "kmeans_cluster",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_ELBOW,
            "k_min": 2,
            "k_max": 4,
            "output_mode": OUTPUT_MODE_FULL,
            "maxiter": 100,
            "tol": 0.0001,
        }
    )
    monkeypatch.setattr(
        creator,
        "_fit_model",
        lambda *, data, agg_client: SimpleNamespace(
            labels_=[0, 0, 0, 0, 1, 1, 1, 1, 1],
            cluster_counts_=[4, 5],
            n_clusters=2,
        ),
    )

    _, transformed_metadata = creator.transform_data_and_metadata(
        data=_data_three_clusters(),
        metadata=_metadata(),
        agg_client=_agg_client(),
    )

    assert transformed_metadata["kmeans_cluster"]["enumerations"] == {
        "cluster_0": "cluster_0",
        "cluster_1": "cluster_1",
    }


def test_transform_data_full_mode_creates_cluster_categorical_column(monkeypatch):
    from exaflow.worker import config as worker_config

    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 1)
    creator = _creator(
        {
            "code": "kmeans_cluster",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_FULL,
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    transformed, transformed_metadata = creator.transform_data_and_metadata(
        data=_data_three_clusters(),
        metadata=_metadata(),
        agg_client=_agg_client(),
    )

    assert transformed.index.tolist() == [10, 11, 12, 20, 21, 22, 30, 31, 32]
    assert "kmeans_cluster" in transformed
    assert set(transformed["kmeans_cluster"]).issubset(
        {"cluster_0", "cluster_1", "cluster_2"}
    )
    assert transformed["kmeans_cluster"].nunique() == 3
    assert transformed_metadata["kmeans_cluster"]["is_categorical"] is True


def test_transform_data_full_mode_supports_multi_start_initialization(monkeypatch):
    from exaflow.worker import config as worker_config

    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 1)
    creator = _creator(
        {
            "code": "kmeans_cluster",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_FULL,
            "init_method": INIT_MULTI_START_RANDOM_RANGE,
            "n_init": 3,
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    transformed, transformed_metadata = creator.transform_data_and_metadata(
        data=_data_three_clusters(),
        metadata=_metadata(),
        agg_client=_agg_client(),
    )

    assert transformed["kmeans_cluster"].nunique() == 3
    assert transformed_metadata["kmeans_cluster"]["is_categorical"] is True


def test_transform_data_binary_mode_creates_yes_no_column(monkeypatch):
    from exaflow.worker import config as worker_config

    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 1)
    creator = _creator(
        {
            "code": "is_cluster_0",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_BINARY,
            "binary_cluster": "cluster_0",
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    transformed, transformed_metadata = creator.transform_data_and_metadata(
        data=_data_three_clusters(),
        metadata=_metadata(),
        agg_client=_agg_client(),
    )

    assert set(transformed["is_cluster_0"]) == {"yes", "no"}
    assert transformed_metadata["is_cluster_0"]["enumerations"] == {
        "yes": "yes",
        "no": "no",
    }


def test_subset_with_one_selected_cluster_is_automatic_binary(monkeypatch):
    from exaflow.worker import config as worker_config

    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 1)
    creator = _creator(
        {
            "code": "auto_binary_cluster",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_SUBSET,
            "selected_clusters": ["cluster_0"],
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    transformed, transformed_metadata = creator.transform_data_and_metadata(
        data=_data_three_clusters(),
        metadata=_metadata(),
        agg_client=_agg_client(),
    )

    assert set(transformed["auto_binary_cluster"]) == {"yes", "no"}
    assert transformed_metadata["auto_binary_cluster"]["enumerations"] == {
        "yes": "yes",
        "no": "no",
    }


def test_subset_with_multiple_clusters_uses_other(monkeypatch):
    from exaflow.worker import config as worker_config

    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 1)
    creator = _creator(
        {
            "code": "selected_clusters",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_SUBSET,
            "selected_clusters": ["cluster_0", "cluster_1"],
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    transformed, transformed_metadata = creator.transform_data_and_metadata(
        data=_data_three_clusters(),
        metadata=_metadata(),
        agg_client=_agg_client(),
    )

    assert set(transformed["selected_clusters"]) == {
        "cluster_0",
        "cluster_1",
        "other",
    }
    assert transformed_metadata["selected_clusters"]["enumerations"] == {
        "cluster_0": "cluster_0",
        "cluster_1": "cluster_1",
        "other": "other",
    }


def test_validate_params_rejects_categorical_cluster_variables():
    creator = _creator(
        {
            "code": "kmeans_cluster",
            "cluster_variables": ["age"],
            "output_mode": OUTPUT_MODE_FULL,
            "maxiter": 100,
            "tol": 0.0001,
        }
    )
    metadata = _metadata()
    metadata["age"]["is_categorical"] = True

    with pytest.raises(BadUserInput, match="must be numerical"):
        creator.validate_params(inputdata=_inputdata(), metadata=metadata)


def test_validate_params_rejects_cluster_id_outside_manual_k():
    creator = _creator(
        {
            "code": "is_cluster_4",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_BINARY,
            "binary_cluster": "cluster_4",
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    with pytest.raises(BadUserInput, match="outside the fitted KMeans cluster range"):
        creator.validate_params(inputdata=_inputdata(), metadata=_metadata())


def test_validate_params_rejects_duplicate_selected_clusters():
    creator = _creator(
        {
            "code": "selected_clusters",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_SUBSET,
            "selected_clusters": ["cluster_1", "cluster_1"],
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    with pytest.raises(BadUserInput, match="should not contain duplicates"):
        creator.validate_params(inputdata=_inputdata(), metadata=_metadata())


def test_validate_params_rejects_invalid_initialization_config():
    creator = _creator(
        {
            "code": "kmeans_cluster",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_FULL,
            "init_method": "kmeans++",
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    with pytest.raises(BadUserInput, match="init_method"):
        creator.validate_params(inputdata=_inputdata(), metadata=_metadata())

    creator = _creator(
        {
            "code": "kmeans_cluster",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_FULL,
            "init_method": INIT_MULTI_START_RANDOM_RANGE,
            "n_init": 0,
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    with pytest.raises(BadUserInput, match="n_init"):
        creator.validate_params(inputdata=_inputdata(), metadata=_metadata())


def test_binary_privacy_validation_rejects_small_yes_or_no_class(monkeypatch):
    from exaflow.worker import config as worker_config

    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 4)
    creator = _creator(
        {
            "code": "is_cluster_0",
            "cluster_variables": ["age", "crp"],
            "k_selection": K_SELECTION_MANUAL,
            "k": 3,
            "output_mode": OUTPUT_MODE_BINARY,
            "binary_cluster": "cluster_0",
            "maxiter": 100,
            "tol": 0.0001,
        }
    )

    with pytest.raises(BadUserInput, match="binary KMeans cluster variable"):
        creator.transform_data_and_metadata(
            data=_data_three_clusters(),
            metadata=_metadata(),
            agg_client=_agg_client(),
        )
