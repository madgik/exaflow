import pandas as pd
import pytest

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.worker.exareme3.udf import udf_service
from exaflow.worker_communication import InsufficientDataError


class _DummyAggClient:
    def __init__(self):
        self.unregistered = False
        self.closed = False

    def unregister(self):
        self.unregistered = True

    def close(self):
        self.closed = True


class _BaseStep:
    def __init__(self, *, params):
        self._params = params

    @classmethod
    def aggregation_server_required(cls):
        return False


class _DropAllStep(_BaseStep):
    def transform_data_and_metadata(self, *, data, metadata, agg_client=None):
        return data.iloc[0:0], metadata


class _FirstOrderedStep(_BaseStep):
    def transform_data_and_metadata(self, *, data, metadata, agg_client=None):
        self._params["calls"].append("first")
        return data, metadata


class _SecondOrderedStep(_BaseStep):
    def transform_data_and_metadata(self, *, data, metadata, agg_client=None):
        self._params["calls"].append("second")
        return data, metadata


class _AggAwareStep(_BaseStep):
    @classmethod
    def aggregation_server_required(cls):
        return True

    def transform_data_and_metadata(self, *, data, metadata, agg_client):
        self._params["received_client"].append(agg_client)
        return data, metadata


class _BaseTransformAggStep(PreprocessingStep):
    def __init__(self, *, params):
        super().__init__(params=params)

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name="base_transform_agg",
            desc="test",
            documentation="test",
            label="test",
            enabled=True,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def validate_params(self, *, inputdata, metadata):
        pass

    def transform_variables(self, *, variables):
        return variables

    def transform_metadata(self, *, metadata):
        return metadata

    def transform_data(self, *, data, agg_client):
        self._params["received_client"].append(agg_client)
        return data


class _NoAggSpecStep(_BaseTransformAggStep):
    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name="no_agg",
            desc="test",
            documentation="test",
            label="test",
            enabled=True,
        )


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


def test_preprocessing_step_base_detects_aggregation_server_component():
    assert _BaseTransformAggStep.aggregation_server_required() is True
    assert _NoAggSpecStep.aggregation_server_required() is False


def test_apply_preprocessing_passes_agg_client_only_to_steps_that_accept_it(
    monkeypatch,
):
    received_client = []
    agg_client = _DummyAggClient()
    monkeypatch.setattr(
        udf_service,
        "exareme3_preprocessing_step_classes",
        {
            "first": _FirstOrderedStep,
            "agg_aware": _AggAwareStep,
        },
    )

    udf_service._apply_preprocessing_steps_to_data_and_metadata(
        data=pd.DataFrame({"x": [1, 2]}),
        metadata={"x": {"is_categorical": False}},
        preprocessing=[
            {"name": "first", "parameters": {"calls": []}},
            {"name": "agg_aware", "parameters": {"received_client": received_client}},
        ],
        check_min_rows=False,
        agg_client=agg_client,
    )

    assert received_client == [agg_client]


def test_base_transform_data_and_metadata_passes_agg_client_to_transform_data():
    received_client = []
    agg_client = _DummyAggClient()
    step = _BaseTransformAggStep(params={"received_client": received_client})

    data, _ = step.transform_data_and_metadata(
        data=pd.DataFrame({"x": [1, 2]}),
        metadata={"x": {"is_categorical": False}},
        agg_client=agg_client,
    )

    assert list(data["x"]) == [1, 2]
    assert received_client == [agg_client]


def test_preprocessing_requirement_can_create_client_for_non_aggregation_udf(
    monkeypatch,
):
    created_clients = []

    class _CreatedAggClient(_DummyAggClient):
        def __init__(self, request_id, aggregator_dns=None):
            super().__init__()
            self.request_id = request_id
            self.aggregator_dns = aggregator_dns
            created_clients.append(self)

    monkeypatch.setattr(udf_service, "AggregationClient", _CreatedAggClient)
    monkeypatch.setattr(
        udf_service,
        "exareme3_preprocessing_step_classes",
        {"agg_aware": _AggAwareStep},
    )

    client = udf_service._create_aggregation_client_if_required(
        request_id="req",
        udf_registry_key="non_agg_udf",
        preprocessing=[{"name": "agg_aware", "parameters": {}}],
    )

    assert client is created_clients[0]
    assert client.request_id == "req"


def test_execute_udf_does_not_pass_preprocessing_only_agg_client_to_udf():
    def udf(data):
        return {"rows": len(data)}

    result = udf_service._execute_udf(
        udf=udf,
        kw_args={},
        data=pd.DataFrame({"x": [1, 2]}),
        metadata={"x": {"is_categorical": False}},
        agg_client=_DummyAggClient(),
        inject_agg_client=False,
    )

    assert result == {"rows": 2}


def test_execute_udf_passes_fixed_agg_client_name_to_aggregation_udf():
    agg_client = _DummyAggClient()

    def udf(data, agg_client):
        return {"rows": len(data), "agg_client": agg_client}

    result = udf_service._execute_udf(
        udf=udf,
        kw_args={},
        data=pd.DataFrame({"x": [1, 2]}),
        metadata={"x": {"is_categorical": False}},
        agg_client=agg_client,
        inject_agg_client=True,
    )

    assert result == {"rows": 2, "agg_client": agg_client}
