import pandas as pd
import pytest

from exaflow.algorithms.exareme3.linear_model.linear_regression import (
    local_step as linear_regression_local_step,
)
from exaflow.algorithms.exareme3.preprocessing.kmeans_cluster_creator import (
    KMeansClusterCreator,
)
from exaflow.worker.exareme3.udf import udf_service
from exaflow.worker_communication import InsufficientDataError
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


class _DummyAggClient:
    def __init__(self):
        self.unregistered = False
        self.closed = False

    def unregister(self):
        self.unregistered = True

    def close(self):
        self.closed = True


class _DropAllStep:
    def __init__(self, *, params):
        self._params = params

    def transform_data_and_metadata(self, *, data, metadata):
        return data.iloc[0:0], metadata


class _FirstOrderedStep:
    def __init__(self, *, params):
        self._params = params

    def transform_data_and_metadata(self, *, data, metadata):
        self._params["calls"].append("first")
        return data, metadata


class _SecondOrderedStep:
    def __init__(self, *, params):
        self._params = params

    def transform_data_and_metadata(self, *, data, metadata):
        self._params["calls"].append("second")
        return data, metadata


def test_check_min_rows_uses_pandas_row_count(monkeypatch):
    monkeypatch.setattr(udf_service.worker_config.privacy, "minimum_row_count", 3)
    data = pd.DataFrame({"x": [1, 2]})
    agg_client = _DummyAggClient()

    with pytest.raises(InsufficientDataError, match="minimum required is 3"):
        udf_service._check_min_rows_or_raise(
            data=data,
            check_min_rows=True,
            agg_client=agg_client,
        )

    assert agg_client.unregistered is True
    assert agg_client.closed is True


def test_apply_preprocessing_checks_min_rows_after_each_step(monkeypatch):
    monkeypatch.setattr(udf_service.worker_config.privacy, "minimum_row_count", 1)
    monkeypatch.setattr(
        udf_service,
        "exareme3_preprocessing_step_classes",
        {"drop_all": _DropAllStep},
    )
    data = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(InsufficientDataError, match="minimum required is 1"):
        udf_service._apply_preprocessing_steps_to_data_and_metadata(
            data=data,
            metadata={"x": {"is_categorical": False}},
            preprocessing=[{"name": "drop_all", "parameters": {}}],
            check_min_rows=True,
            agg_client=None,
        )


def test_apply_preprocessing_checks_min_rows_when_no_steps(monkeypatch):
    monkeypatch.setattr(udf_service.worker_config.privacy, "minimum_row_count", 3)
    data = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(InsufficientDataError, match="minimum required is 3"):
        udf_service._apply_preprocessing_steps_to_data_and_metadata(
            data=data,
            metadata={"x": {"is_categorical": False}},
            preprocessing=[],
            check_min_rows=True,
            agg_client=None,
        )


def test_apply_preprocessing_uses_request_order(monkeypatch):
    calls = []
    monkeypatch.setattr(
        udf_service,
        "exareme3_preprocessing_step_classes",
        {"first": _FirstOrderedStep, "second": _SecondOrderedStep},
    )

    udf_service._apply_preprocessing_steps_to_data_and_metadata(
        data=pd.DataFrame({"x": [1, 2]}),
        metadata={"x": {"is_categorical": False}},
        preprocessing=[
            {"name": "second", "parameters": {"calls": calls}},
            {"name": "first", "parameters": {"calls": calls}},
        ],
        check_min_rows=False,
        agg_client=None,
    )

    assert calls == ["second", "first"]


def test_kmeans_cluster_creator_runtime_output_feeds_linear_regression(monkeypatch):
    monkeypatch.setattr(udf_service.worker_config.privacy, "minimum_row_count", 1)
    monkeypatch.setattr(
        udf_service,
        "exareme3_preprocessing_step_classes",
        {"kmeans_cluster_creator": KMeansClusterCreator},
    )
    agg_client = SimulatedAggClient(
        worker_id=0,
        coordinator=AggregationCoordinator(n_workers=1),
    )
    data = pd.DataFrame(
        {
            "age": [40.0, 41.0, 42.0, 70.0, 71.0, 72.0],
            "crp": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
            "outcome": [10.0, 10.5, 11.0, 20.0, 20.5, 21.0],
        }
    )
    metadata = {
        "age": {"is_categorical": False, "sql_type": "real"},
        "crp": {"is_categorical": False, "sql_type": "real"},
        "outcome": {"is_categorical": False, "sql_type": "real"},
    }

    transformed_data, transformed_metadata = (
        udf_service._apply_preprocessing_steps_to_data_and_metadata(
            data=data,
            metadata=metadata,
            preprocessing=[
                {
                    "name": "kmeans_cluster_creator",
                    "parameters": {
                        "code": "kmeans_cluster",
                        "cluster_variables": ["age", "crp"],
                        "k_selection": "manual",
                        "k": 2,
                        "output_mode": "full",
                        "maxiter": 100,
                        "tol": 0.0001,
                    },
                }
            ],
            check_min_rows=False,
            agg_client=agg_client,
        )
    )

    assert transformed_metadata["kmeans_cluster"]["is_categorical"] is True
    assert set(transformed_data["kmeans_cluster"]) == {"cluster_0", "cluster_1"}

    result = linear_regression_local_step(
        agg_client=agg_client,
        data=transformed_data,
        y_var="outcome",
        x_vars=["kmeans_cluster"],
        categorical_vars=["kmeans_cluster"],
        numerical_vars=[],
    )

    assert result["n_obs"] == len(transformed_data)
    assert any("kmeans_cluster" in name for name in result["feature_names"])


def test_execute_udf_does_not_pass_preprocessing_agg_client_to_plain_udf():
    def plain_udf(*, data, metadata):
        return {
            "columns": list(data.columns),
            "metadata_keys": list(metadata),
        }

    data = pd.DataFrame({"x": [1, 2]})
    metadata = {"x": {"is_categorical": False}}

    result = udf_service._execute_udf(
        udf=plain_udf,
        kw_args={},
        data=data,
        metadata=metadata,
        agg_client=_DummyAggClient(),
    )

    assert result == {"columns": ["x"], "metadata_keys": ["x"]}


def test_execute_udf_passes_agg_client_when_udf_accepts_it():
    agg_client = _DummyAggClient()

    def aggregation_udf(*, data, agg_client):
        return agg_client

    result = udf_service._execute_udf(
        udf=aggregation_udf,
        kw_args={},
        data=pd.DataFrame({"x": [1, 2]}),
        metadata={"x": {"is_categorical": False}},
        agg_client=agg_client,
    )

    assert result is agg_client


def test_execute_udf_uses_registry_aggregation_client_name():
    agg_client = _DummyAggClient()

    def aggregation_udf(*, data, client):
        return client

    result = udf_service._execute_udf(
        udf=aggregation_udf,
        kw_args={},
        data=pd.DataFrame({"x": [1, 2]}),
        metadata={"x": {"is_categorical": False}},
        agg_client=agg_client,
        agg_client_name="client",
    )

    assert result is agg_client
