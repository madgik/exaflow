"""Exercise parameter-driven output widths through API and worker boundaries."""

import pandas as pd
import pytest
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import ParseDict
from pydantic import ValidationError

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.utils.output_specifications import resolve_output_metadata
from exaflow.controller.services.api import analysis_request_validator
from exaflow.controller.services.api.algorithm_spec_dtos import (
    _convert_transformer_specification_to_dto,
)
from exaflow.controller.services.api.analysis_request_dtos import (
    AnalysisPreprocessingStepDTO,
)
from exaflow.protos.worker import worker_pb2
from exaflow.worker.exareme3.udf import udf_service
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.test_preprocessing_output_specifications import (
    TwoColumnProjection,
)
from tests.standalone_tests.test_preprocessing_output_specifications import _cdes
from tests.standalone_tests.test_preprocessing_output_specifications import _parameter
from tests.standalone_tests.test_preprocessing_output_specifications import _request
from tests.standalone_tests.test_preprocessing_output_specifications import _spec

NUMERICAL = {"type": "real", "stattype": "numerical"}
DYNAMIC = {**NUMERICAL, "column_count_param": "n_components"}


def _dynamic_spec(columns=None, *, count_parameter=None):
    return _spec(
        [DYNAMIC] if columns is None else columns,
        n_components=(
            _parameter(specs.ParameterType.INT)
            if count_parameter is None
            else count_parameter
        ),
    )


class DynamicProjection(TwoColumnProjection):
    """Test projection with fixed outputs surrounding a dynamic component block."""

    @classmethod
    def get_specification(cls):
        return _dynamic_spec([NUMERICAL, DYNAMIC, NUMERICAL])

    def transform_data(self, *, data):
        return data.assign(
            **{
                name: data["age"].astype(float) * (index + 1)
                for index, name in enumerate(self._params["new_column_names"])
            }
        )


@pytest.mark.parametrize("count", [1, 3])
def test_mixed_outputs_agree_across_controller_and_worker_protobuf(monkeypatch, count):
    monkeypatch.setitem(
        analysis_request_validator.exareme3_preprocessing_step_classes,
        "test_projection",
        DynamicProjection,
    )
    names = ["before", *[f"pca_{index}" for index in range(1, count + 1)], "after"]
    parameters = {"new_column_names": names, "n_components": count}
    step = AnalysisPreprocessingStepDTO(name="test_projection", parameters=parameters)
    inputdata, cdes = analysis_request_validator._validate_and_apply_preprocessing(
        _request(preprocessing=[step]),
        {"test_projection": DynamicProjection.get_specification()},
        _cdes(),
    )

    request = worker_pb2.RunUdfRequest()
    ParseDict({"preprocessing": [step.model_dump(mode="json")]}, request.system_args)
    transported = MessageToDict(request.system_args, preserving_proto_field_name=True)
    assert type(transported["preprocessing"][0]["parameters"]["n_components"]) is float
    data, metadata = udf_service._apply_preprocessing_steps_to_data_and_metadata(
        data=pd.DataFrame({"age": [20, 40]}),
        metadata={"age": _cdes()["age"].model_dump()},
        preprocessing=transported["preprocessing"],
        check_min_rows=False,
        agg_client=None,
    )
    assert inputdata.variables[-len(names) :] == list(data.columns[1:]) == names
    assert data.loc[0, names].tolist() == [
        20.0 * index for index in range(1, len(names) + 1)
    ]
    for name in names:
        assert cdes[name].sql_type == metadata[name]["sql_type"] == "real"
        assert not cdes[name].is_categorical
        assert cdes[name].enumerations is None


@pytest.mark.parametrize("count", [1, 4])
def test_dynamic_only_outputs(count):
    names = [f"pca_{index}" for index in range(1, count + 1)]
    resolved = resolve_output_metadata(
        _dynamic_spec(),
        {
            "n_components": count,
            "new_column_names": names,
        },
        {},
    )
    assert list(resolved) == names
    assert all(column["sql_type"] == "real" for column in resolved.values())


def test_multiple_dynamic_groups_and_independent_nominal_metadata():
    nominal = {
        "type": "text",
        "stattype": "nominal",
        "enumerations": {"source_parameter_name": "rules", "source_type": "dict_keys"},
    }
    specification = _spec(
        [nominal, {**nominal, "column_count_param": "group_count"}, DYNAMIC, NUMERICAL],
        group_count=_parameter(specs.ParameterType.INT),
        n_components=_parameter(specs.ParameterType.INT),
        rules=_parameter(specs.ParameterType.DICT),
    )
    names = ["group", "group_a", "group_b", "pca_1", "pca_2", "pca_3", "after"]
    resolved = resolve_output_metadata(
        specification,
        {
            "new_column_names": names,
            "group_count": 2,
            "n_components": 3,
            "rules": {"low": {}, "high": {}},
        },
        {},
    )
    assert list(resolved) == names
    assert [column["is_categorical"] for column in resolved.values()] == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    for name in names[:3]:
        assert resolved[name]["enumerations"] == {"low": "low", "high": "high"}
    resolved["group_a"]["enumerations"]["extra"] = "extra"
    assert "extra" not in resolved["group_b"]["enumerations"]
    assert "extra" not in resolved["group"]["enumerations"]


@pytest.mark.parametrize("minimum", [None, 1])
def test_count_parameter_min_is_optional_and_dto_preserves_dynamic_reference(minimum):
    parameter = _parameter(specs.ParameterType.INT)
    if minimum is not None:
        parameter = parameter.model_copy(update={"min": minimum})
    specification = _dynamic_spec([NUMERICAL, DYNAMIC], count_parameter=parameter)
    dto = _convert_transformer_specification_to_dto(specification)
    columns = dto.model_dump(mode="json")["outputs"]["columns"]
    assert columns == [
        {**NUMERICAL, "column_count_param": None, "enumerations": None},
        {**DYNAMIC, "enumerations": None},
    ]
    assert dto.parameters["n_components"].min == minimum


@pytest.mark.parametrize("reference", ["", " ", "\t"])
def test_rejects_blank_count_parameter_name(reference):
    with pytest.raises(ValidationError):
        _dynamic_spec([{**NUMERICAL, "column_count_param": reference}])


@pytest.mark.parametrize(
    "parameter",
    [
        _parameter(specs.ParameterType.REAL),
        _parameter(specs.ParameterType.TEXT),
        _parameter(specs.ParameterType.INT, required=False),
        _parameter(specs.ParameterType.TEXT, multiple=True, required=False),
    ],
)
def test_rejects_invalid_count_parameter_contract(parameter):
    with pytest.raises(ValidationError, match="n_components"):
        _dynamic_spec(count_parameter=parameter)


def test_rejects_missing_parameter_reference_and_literal_count():
    with pytest.raises(ValidationError, match="n_components"):
        _spec([DYNAMIC])
    with pytest.raises(ValidationError, match="column_count"):
        _dynamic_spec([{**NUMERICAL, "column_count": 2}])


@pytest.mark.parametrize(
    "count",
    [
        None,
        True,
        False,
        0,
        -1,
        0.0,
        -1.0,
        1.5,
        "2",
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_rejects_invalid_dynamic_count(count):
    with pytest.raises(BadUserInput, match="n_components.*positive integer"):
        resolve_output_metadata(
            _dynamic_spec(),
            {
                "new_column_names": ["pca_1"],
                "n_components": count,
            },
            {},
        )


def test_missing_required_dynamic_count_does_not_use_parameter_default():
    parameter = _parameter(specs.ParameterType.INT).model_copy(update={"default": 1})
    with pytest.raises(BadUserInput, match="n_components.*positive integer"):
        resolve_output_metadata(
            _dynamic_spec(count_parameter=parameter),
            {
                "new_column_names": ["pca_1"],
            },
            {},
        )


@pytest.mark.parametrize("names", [["pca_1"], ["pca_1", "pca_2", "pca_3"]])
def test_rejects_expanded_name_count_mismatch(names):
    with pytest.raises(BadUserInput, match="exactly 2"):
        resolve_output_metadata(
            _dynamic_spec(),
            {
                "new_column_names": names,
                "n_components": 2,
            },
            {},
        )


def test_validates_name_count_before_expanding_large_count():
    with pytest.raises(BadUserInput, match="nonblank column names"):
        resolve_output_metadata(
            _dynamic_spec(),
            {
                "new_column_names": ["pca_1"],
                "n_components": 10**30,
            },
            {},
        )


@pytest.mark.parametrize(
    "names,existing",
    [
        (["a", "b", "a"], {}),
        (["a", "b", "age"], {"age": {}}),
        (["a", "", "c"], {}),
        (["a", " ", "c"], {}),
    ],
)
def test_rejects_invalid_names_across_fixed_and_dynamic_blocks(names, existing):
    with pytest.raises(BadUserInput):
        resolve_output_metadata(
            _dynamic_spec([NUMERICAL, DYNAMIC]),
            {
                "new_column_names": names,
                "n_components": 2,
            },
            existing,
        )


def test_public_api_keeps_integer_type_validation(monkeypatch):
    monkeypatch.setitem(
        analysis_request_validator.exareme3_preprocessing_step_classes,
        "test_projection",
        DynamicProjection,
    )
    with pytest.raises(BadUserInput):
        analysis_request_validator._validate_and_apply_preprocessing(
            _request(
                preprocessing=[
                    AnalysisPreprocessingStepDTO(
                        name="test_projection",
                        parameters={
                            "n_components": 2.0,
                            "new_column_names": ["before", "pca_1", "pca_2", "after"],
                        },
                    )
                ]
            ),
            {"test_projection": DynamicProjection.get_specification()},
            _cdes(),
        )


def _list_spec(columns=None, *, parameter=None):
    return _spec(
        [{**NUMERICAL, "column_count_param": "selected_items"}]
        if columns is None
        else columns,
        selected_items=(
            _parameter(specs.ParameterType.TEXT, multiple=True)
            if parameter is None
            else parameter
        ),
    )


@pytest.mark.parametrize("count", [1, 2, 4])
def test_list_length_defines_output_count_and_survives_dto(count):
    specification = _list_spec()
    parameters = {
        "selected_items": [f"item_{i}" for i in range(count)],
        "new_column_names": [f"derived_{i}" for i in range(count)],
    }
    dto = _convert_transformer_specification_to_dto(specification)
    assert (
        dto.model_dump(mode="json")["outputs"]["columns"][0]["column_count_param"]
        == "selected_items"
    )
    assert dto.parameters["selected_items"].multiple
    resolved = resolve_output_metadata(specification, parameters, {})
    assert list(resolved) == parameters["new_column_names"]
    assert all(column["sql_type"] == "real" for column in resolved.values())


@pytest.mark.parametrize("type_", [specs.ParameterType.TEXT, specs.ParameterType.INT])
def test_required_list_count_is_based_on_cardinality_not_element_values(type_):
    specification = _list_spec(parameter=_parameter(type_, multiple=True))
    items = ["second", "first"] if type_ == specs.ParameterType.TEXT else [100, -1]
    assert list(
        resolve_output_metadata(
            specification, {"selected_items": items, "new_column_names": ["a", "b"]}, {}
        )
    ) == ["a", "b"]


@pytest.mark.parametrize("items", [None, [], "ab", 2, True, ("a", "b"), {}])
def test_list_count_rejects_missing_empty_or_nonlist_values(items):
    with pytest.raises(BadUserInput, match="selected_items.*nonempty list"):
        resolve_output_metadata(
            _list_spec(), {"selected_items": items, "new_column_names": ["a"]}, {}
        )


def test_required_list_count_does_not_fall_back_to_a_default():
    parameter = _parameter(specs.ParameterType.TEXT, multiple=True).model_copy(
        update={"default": ["item"]}
    )
    with pytest.raises(BadUserInput, match="selected_items.*nonempty list"):
        resolve_output_metadata(
            _list_spec(parameter=parameter), {"new_column_names": ["a"]}, {}
        )


@pytest.mark.parametrize("names", [["a"], ["a", "b", "c"]])
def test_list_count_rejects_source_output_length_mismatch(names):
    with pytest.raises(BadUserInput, match="exactly 2"):
        resolve_output_metadata(
            _list_spec(), {"selected_items": ["x", "y"], "new_column_names": names}, {}
        )


def test_list_and_integer_counts_can_be_mixed_with_fixed_outputs():
    specification = _spec(
        [NUMERICAL, {**NUMERICAL, "column_count_param": "selected_items"}, DYNAMIC],
        selected_items=_parameter(specs.ParameterType.TEXT, multiple=True),
        n_components=_parameter(specs.ParameterType.INT),
    )
    names = ["fixed", "left", "right", "component"]
    resolved = resolve_output_metadata(
        specification,
        {"selected_items": ["x", "y"], "n_components": 1, "new_column_names": names},
        {},
    )
    assert list(resolved) == names
    assert resolved["left"] is not resolved["right"]


class ListCountProjection(DynamicProjection):
    @classmethod
    def get_specification(cls):
        return _list_spec()


@pytest.mark.parametrize("count", [1, 3])
def test_list_count_agrees_across_controller_and_worker_protobuf(monkeypatch, count):
    monkeypatch.setitem(
        analysis_request_validator.exareme3_preprocessing_step_classes,
        "test_projection",
        ListCountProjection,
    )
    names = [f"derived_{i}" for i in range(count)]
    step = AnalysisPreprocessingStepDTO(
        name="test_projection",
        parameters={
            "selected_items": [f"item_{i}" for i in range(count)],
            "new_column_names": names,
        },
    )
    inputdata, cdes = analysis_request_validator._validate_and_apply_preprocessing(
        _request(preprocessing=[step]),
        {"test_projection": ListCountProjection.get_specification()},
        _cdes(),
    )
    request = worker_pb2.RunUdfRequest()
    ParseDict({"preprocessing": [step.model_dump(mode="json")]}, request.system_args)
    transported = MessageToDict(request.system_args, preserving_proto_field_name=True)
    data, metadata = udf_service._apply_preprocessing_steps_to_data_and_metadata(
        data=pd.DataFrame({"age": [20, 40]}),
        metadata={"age": _cdes()["age"].model_dump()},
        preprocessing=transported["preprocessing"],
        check_min_rows=False,
        agg_client=None,
    )
    assert inputdata.variables[-count:] == list(data.columns[1:]) == names
    for index, name in enumerate(names):
        assert cdes[name].sql_type == metadata[name]["sql_type"] == "real"
        assert not cdes[name].is_categorical
        assert metadata[name]["is_categorical"] is False
        assert data[name].tolist() == [20.0 * (index + 1), 40.0 * (index + 1)]
