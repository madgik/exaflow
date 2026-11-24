"""StandardScaler registration, public contract and worker-side execution."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import ParseDict
from sklearn.preprocessing import StandardScaler as ReferenceScaler

from exaflow import exareme3_preprocessing_step_classes
from exaflow.algorithms.exareme3.feature_engineering.standard_scaler import (
    StandardScalerStep,
)
from exaflow.algorithms.federated.statistics.describe import FederatedDescribe
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.controller.services.api.algorithm_spec_dtos import (
    _convert_transformer_specification_to_dto,
)
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.api.analysis_request_validator import (
    _validate_and_apply_preprocessing,
)
from exaflow.controller.services.api.analysis_request_validator import (
    _validate_parameters,
)
from exaflow.protos.worker import worker_pb2
from exaflow.worker import config as worker_config
from exaflow.worker.exareme3.udf import udf_service
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement
from exaflow.worker_communication import InsufficientDataError
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def metadata():
    return {
        "crp": {
            "code": "crp",
            "label": "CRP",
            "sql_type": "real",
            "is_categorical": False,
            "min": 0,
            "max": 100,
        }
    }


def parameters(**overrides):
    return {
        "source_variables": ["crp"],
        "new_column_names": ["crp__scaled"],
        **overrides,
    }


def inputdata():
    return Inputdata(data_model="test:0.1", datasets=["test"], variables=["crp"])


def validate_request(params, *, variables=None, source_metadata=None):
    variables = ["crp"] if variables is None else variables
    source_metadata = metadata() if source_metadata is None else source_metadata
    request = AnalysisRequestDTO(
        inputdata={
            "data_model": "test:0.1",
            "datasets": ["test"],
            "variables": variables,
        },
        preprocessing=[{"name": "standard_scaler", "parameters": params}],
        algorithm={"name": "describe", "x": [], "y": variables, "parameters": {}},
    )
    return _validate_and_apply_preprocessing(
        request,
        {"standard_scaler": StandardScalerStep.get_specification()},
        {name: CommonDataElement(**value) for name, value in source_metadata.items()},
    )


@pytest.fixture(autouse=True)
def fixed_privacy_minimum(monkeypatch):
    monkeypatch.setattr(worker_config.privacy, "minimum_row_count", 10)


class CountingClient(DummyAggClient):
    calls = 0
    unregistered = False

    def sum(self, value):
        self.calls += 1
        return super().sum(value)

    def unregister(self):
        self.unregistered = True


def test_registration_and_inline_specification():
    assert exareme3_preprocessing_step_classes["standard_scaler"] is StandardScalerStep
    spec = StandardScalerStep.get_specification()
    assert set(spec.parameters) == {"source_variables", "new_column_names"}
    assert spec.parameters["source_variables"].multiple
    assert spec.outputs.new_column_names_param == "new_column_names"
    assert len(spec.outputs.columns) == 1
    assert spec.outputs.columns[0].type.value == "real"
    assert spec.outputs.columns[0].stattype.value == "numerical"
    assert spec.outputs.columns[0].column_count_param == "source_variables"
    assert StandardScalerStep.aggregation_server_required()


def test_specification_dto_and_normal_request_validation():
    spec = StandardScalerStep.get_specification()
    serialized = _convert_transformer_specification_to_dto(spec).model_dump(mode="json")
    assert serialized["name"] == "standard_scaler"
    assert set(serialized["parameters"]) == {"source_variables", "new_column_names"}
    assert serialized["outputs"]["columns"][0]["type"] == "real"
    _validate_parameters(
        parameters=parameters(),
        parameters_specs=spec.parameters,
        inputdata=inputdata(),
        data_model_cdes={},
    )


@pytest.mark.parametrize("partition_at", [0, 13, 40])
def test_global_fit_preserves_source_and_metadata(partition_at):
    data = pd.DataFrame({"crp": np.linspace(2, 20, 40)})
    data.loc[2, "crp"] = np.nan
    data["dataset"] = "test"
    original = data.copy(deep=True)
    original_metadata = metadata()
    coordinator = AggregationCoordinator(2)

    def run_worker(worker_id, part):
        step = StandardScalerStep(params=parameters())
        step.validate_params(inputdata=inputdata(), metadata=original_metadata)
        try:
            return step.transform_data_and_metadata(
                data=part.copy(),
                metadata=original_metadata,
                agg_client=SimulatedAggClient(
                    worker_id=worker_id, coordinator=coordinator
                ),
            )
        except Exception as exc:
            coordinator.abort(exc)
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_worker, 0, data.iloc[:partition_at]),
            executor.submit(run_worker, 1, data.iloc[partition_at:]),
        ]
        results = [future.result(timeout=30) for future in futures]
    actual = pd.concat([frame for frame, _ in results])
    np.testing.assert_allclose(
        actual["crp__scaled"],
        ReferenceScaler().fit_transform(data[["crp"]])[:, 0],
        equal_nan=True,
    )
    pd.testing.assert_series_equal(actual["crp"], original["crp"])
    pd.testing.assert_frame_equal(data, original)
    assert original_metadata == metadata()
    for _, result_metadata in results:
        assert result_metadata["crp"] == original_metadata["crp"]
        assert result_metadata["crp__scaled"] == {
            "code": "crp__scaled",
            "label": "crp__scaled",
            "sql_type": "real",
            "is_categorical": False,
        }


@pytest.mark.parametrize("count", [1, 9, 10, 11])
def test_scaler_does_not_add_a_nonmissing_floor_to_worker_row_policy(count):
    frame = pd.DataFrame({"crp": list(np.arange(float(count))) + [np.nan] * 10})
    original = frame.copy(deep=True)
    client = CountingClient()
    result, _ = udf_service._apply_preprocessing_steps_to_data_and_metadata(
        data=frame,
        metadata=metadata(),
        preprocessing=[{"name": "standard_scaler", "parameters": parameters()}],
        check_min_rows=True,
        agg_client=client,
    )
    assert client.calls == 3
    assert not client.unregistered
    pd.testing.assert_series_equal(result["crp"], original["crp"])
    np.testing.assert_allclose(
        result["crp__scaled"],
        ReferenceScaler().fit_transform(original)[:, 0],
        equal_nan=True,
    )


def test_existing_worker_minimum_still_rejects_a_small_participant():
    frame = pd.DataFrame({"crp": np.arange(9.0)})
    original = frame.copy(deep=True)
    client = CountingClient()
    with pytest.raises(InsufficientDataError, match="minimum required is 10"):
        udf_service._apply_preprocessing_steps_to_data_and_metadata(
            data=frame,
            metadata=metadata(),
            preprocessing=[{"name": "standard_scaler", "parameters": parameters()}],
            check_min_rows=True,
            agg_client=client,
        )
    assert client.calls == 0
    assert client.unregistered
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("small_group", ["small", None])
def test_scaler_uses_worker_policy_without_silently_excluding_small_datasets(
    small_group,
):
    # Twelve participant rows pass the common floor even though one dataset has two.
    frame = pd.DataFrame(
        {
            "dataset": [small_group] * 2 + ["large"] * 10,
            "crp": [100.0, 200.0] + list(np.arange(10.0)),
        }
    )
    original = frame.copy(deep=True)
    client = CountingClient()
    result, _ = udf_service._apply_preprocessing_steps_to_data_and_metadata(
        data=frame,
        metadata=metadata(),
        preprocessing=[{"name": "standard_scaler", "parameters": parameters()}],
        check_min_rows=True,
        agg_client=client,
    )
    assert client.calls == 3
    assert len(result) == len(original)
    np.testing.assert_allclose(
        result["crp__scaled"], ReferenceScaler().fit_transform(original[["crp"]])[:, 0]
    )
    pd.testing.assert_frame_equal(result[original.columns], original)


def test_zero_contribution_group_is_allowed_without_dropping_rows():
    frame = pd.DataFrame(
        {
            "dataset": ["missing"] * 10 + ["large"] * 10,
            "crp": [np.nan] * 10 + list(np.arange(10.0)),
        }
    )
    original = frame.copy(deep=True)
    result, _ = StandardScalerStep(params=parameters()).transform_data_and_metadata(
        data=frame, metadata=metadata(), agg_client=DummyAggClient()
    )
    assert len(result) == len(original)
    pd.testing.assert_series_equal(result["crp"], original["crp"])
    np.testing.assert_allclose(
        result["crp__scaled"],
        ReferenceScaler().fit_transform(original[["crp"]])[:, 0],
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "names", [None, [], ["a", "b"], [""], ["  "], ["crp"], "scaled", [1]]
)
def test_output_contract_rejects_invalid_names(names):
    with pytest.raises(BadUserInput):
        validate_request(parameters(new_column_names=names))


@pytest.mark.parametrize(
    "source", [None, "crp", [], ["absent"], [""], [1], ["crp", "crp"]]
)
def test_invalid_source_rejected(source):
    with pytest.raises(BadUserInput):
        validate_request(parameters(source_variables=source))


@pytest.mark.parametrize(
    "updates",
    [
        {"sql_type": "text"},
        {"is_categorical": True},
        {"sql_type": "int", "is_categorical": True},
    ],
)
def test_nonnumerical_source_rejected(updates):
    changed = deepcopy(metadata())
    changed["crp"].update(updates)
    with pytest.raises(BadUserInput, match="numerical"):
        StandardScalerStep(params=parameters()).validate_params(
            inputdata=inputdata(), metadata=changed
        )


def test_integer_constant_source_creates_real_output_without_changing_source():
    source_metadata = metadata()
    source_metadata["crp"]["sql_type"] = "int"
    original_metadata = deepcopy(source_metadata)
    frame = pd.DataFrame({"crp": [5] * 10})
    source = frame["crp"].copy()
    step = StandardScalerStep(params=parameters())
    step.validate_params(inputdata=inputdata(), metadata=source_metadata)
    result, result_metadata = step.transform_data_and_metadata(
        data=frame, metadata=source_metadata, agg_client=DummyAggClient()
    )
    np.testing.assert_array_equal(result["crp__scaled"], np.zeros(10))
    assert pd.api.types.is_float_dtype(result["crp__scaled"])
    pd.testing.assert_series_equal(result["crp"], source)
    assert result_metadata["crp"] == original_metadata["crp"]
    assert result_metadata["crp__scaled"]["sql_type"] == "real"
    assert source_metadata == original_metadata


@pytest.mark.parametrize(
    "extra",
    [
        {"recipe": {}},
        {"minimum_row_count": 1},
        {"source_variable": "crp"},
        {"n_variables": 1},
    ],
)
def test_controller_rejects_unsupported_parameters(extra):
    with pytest.raises(BadUserInput, match="does not exist"):
        _validate_parameters(
            parameters=parameters(**extra),
            parameters_specs=StandardScalerStep.get_specification().parameters,
            inputdata=inputdata(),
            data_model_cdes={},
        )


def test_variables_and_declared_aggregation_requirement():
    step = StandardScalerStep(params=parameters())
    assert step.transform_variables(variables=["crp"]) == ["crp", "crp__scaled"]
    assert udf_service._preprocessing_requires_aggregation_server(
        [{"name": "standard_scaler", "parameters": parameters()}]
    )


def test_real_worker_preprocessing_loop_forwards_derived_column():
    data = pd.DataFrame({"crp": np.linspace(2, 20, 40)})
    original = data["crp"].copy()
    result, result_metadata = (
        udf_service._apply_preprocessing_steps_to_data_and_metadata(
            data=data,
            metadata=metadata(),
            preprocessing=[
                {"name": "standard_scaler", "parameters": parameters()},
                {
                    "name": "standard_scaler",
                    "parameters": parameters(
                        source_variables=["crp__scaled"], new_column_names=["second"]
                    ),
                },
            ],
            check_min_rows=True,
            agg_client=DummyAggClient(),
        )
    )
    np.testing.assert_allclose(result["second"], result["crp__scaled"], atol=1e-14)
    pd.testing.assert_series_equal(result["crp"], original)
    assert list(result.columns) == ["crp", "crp__scaled", "second"]
    assert result_metadata["second"]["sql_type"] == "real"


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("partition_at", [0, 13, 40])
def test_multiple_sources_preserve_order_values_and_empty_participants(
    count, partition_at
):
    frame = pd.DataFrame(
        {
            "crp": np.linspace(2, 20, 40),
            "age": np.arange(40.0) + 100,
            "constant": np.full(40, 5),
            "dataset": ["test"] * 40,
        }
    )
    frame.loc[2, "crp"] = np.nan
    selected = ["crp", "age", "constant"][:count]
    sources = list(reversed(selected))
    names = [f"derived_{i}" for i in range(count)]
    original = frame.copy(deep=True)
    source_metadata = {
        source: {**metadata()["crp"], "code": source, "label": source}
        for source in selected
    }
    params = parameters(source_variables=sources, new_column_names=names)
    inputdata_result, cdes = validate_request(
        params, variables=selected, source_metadata=source_metadata
    )
    assert inputdata_result.variables == selected + names
    coordinator = AggregationCoordinator(2)

    def run_worker(worker_id, part):
        try:
            request = worker_pb2.RunUdfRequest()
            ParseDict(
                {"preprocessing": [{"name": "standard_scaler", "parameters": params}]},
                request.system_args,
            )
            transported = MessageToDict(
                request.system_args, preserving_proto_field_name=True
            )
            return udf_service._apply_preprocessing_steps_to_data_and_metadata(
                data=part.copy(),
                metadata=deepcopy(source_metadata),
                preprocessing=transported["preprocessing"],
                check_min_rows=False,
                agg_client=SimulatedAggClient(
                    worker_id=worker_id, coordinator=coordinator
                ),
            )
        except Exception as exc:
            coordinator.abort(exc)
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_worker, 0, frame.iloc[:partition_at]),
            executor.submit(run_worker, 1, frame.iloc[partition_at:]),
        ]
        results = [future.result(timeout=30) for future in futures]
    actual = pd.concat([result for result, _ in results])
    np.testing.assert_allclose(
        actual[names],
        ReferenceScaler().fit_transform(original[sources]),
        equal_nan=True,
    )
    pd.testing.assert_frame_equal(actual[original.columns], original)
    pd.testing.assert_frame_equal(frame, original)
    assert list(actual.columns) == list(original.columns) + names
    for _, result_metadata in results:
        assert {key: result_metadata[key] for key in selected} == source_metadata
        for name in names:
            assert result_metadata[name]["sql_type"] == cdes[name].sql_type == "real"


def test_controller_checks_output_namespace_including_unselected_dataset():
    full_metadata = {
        **metadata(),
        "dataset": {
            "code": "dataset",
            "label": "Dataset",
            "sql_type": "text",
            "is_categorical": True,
            "enumerations": {"test": "Test"},
        },
    }
    with pytest.raises(BadUserInput, match="already exists"):
        validate_request(
            parameters(new_column_names=["dataset"]), source_metadata=full_metadata
        )


def test_controller_rejects_duplicate_sources_with_matching_output_count():
    with pytest.raises(BadUserInput, match="distinct"):
        validate_request(
            parameters(source_variables=["crp", "crp"], new_column_names=["a", "b"])
        )


def test_describe_retains_its_own_small_group_masking_after_scaling():
    data, _ = udf_service._apply_preprocessing_steps_to_data_and_metadata(
        data=pd.DataFrame({"crp": [42.0], "dataset": ["test"]}),
        metadata=metadata(),
        preprocessing=[{"name": "standard_scaler", "parameters": parameters()}],
        check_min_rows=False,
        agg_client=DummyAggClient(),
    )
    report = FederatedDescribe(DummyAggClient()).describe(
        data=data,
        numerical_vars=["crp", "crp__scaled"],
        nominal_vars=[],
        min_row_count=10,
        nominal_levels={},
    )
    records = report.recs + report.global_recs
    assert {record["variable"] for record in records} == {"crp", "crp__scaled"}
    assert all(record["data"] is None for record in records)
