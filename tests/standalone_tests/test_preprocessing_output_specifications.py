from copy import deepcopy

import pandas as pd
import pytest
from pydantic import ValidationError

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.feature_engineering.categorical_column_creator import (
    CategoricalColumnCreator,
)
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.utils.output_specifications import resolve_output_metadata
from exaflow.algorithms.utils.output_specifications import validate_output_metadata
from exaflow.controller.services.api import analysis_request_validator
from exaflow.controller.services.api.algorithm_spec_dtos import (
    _convert_transformer_specification_to_dto,
)
from exaflow.controller.services.api.analysis_request_dtos import (
    AnalysisPreprocessingStepDTO,
)
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.controller.services.api.test_validate_algorithm_request import (
    _cdes,
)
from tests.standalone_tests.controller.services.api.test_validate_algorithm_request import (
    _request,
)


def _parameter(type_, *, multiple=False, required=True):
    return specs.ParameterSpecification(
        label="Test parameter",
        desc="Test parameter.",
        types=[type_],
        required=required,
        multiple=multiple,
    )


def _spec(columns=None, **parameters):
    return specs.PreprocessingStepSpecification(
        name="test_projection",
        label="Test projection",
        desc="Projects data.",
        documentation="Two numerical columns for exercising the output contract.",
        enabled=True,
        parameters={
            "new_column_names": _parameter(specs.ParameterType.TEXT, multiple=True),
            **parameters,
        },
        outputs={
            "new_column_names_param": "new_column_names",
            "columns": columns
            if columns is not None
            else [
                {"type": "real", "stattype": "numerical"},
                {"type": "real", "stattype": "numerical"},
            ],
        },
    )


def _range_spec():
    return _spec(
        [
            {
                "type": "text",
                "stattype": "nominal",
                "enumerations": {
                    "source_parameter_name": "count",
                    "source_type": "prefixed_range",
                    "prefix_parameter_name": "prefix",
                },
            },
        ],
        count=_parameter(specs.ParameterType.INT),
        prefix=_parameter(specs.ParameterType.TEXT),
    )


class TwoColumnProjection(PreprocessingStep):
    @classmethod
    def get_specification(cls):
        return _spec()

    def validate_params(self, *, inputdata, metadata):
        resolve_output_metadata(self.get_specification(), self._params, metadata)

    def transform_variables(self, *, variables):
        return variables + self._params["new_column_names"]

    def transform_metadata(self, *, metadata):
        return {
            **deepcopy(metadata),
            **resolve_output_metadata(self.get_specification(), self._params, metadata),
        }

    def transform_data(self, *, data):
        first, second = self._params["new_column_names"]
        return data.assign(**{first: data["age"] * 0.5, second: data["age"] * 2.0})


def test_two_numerical_outputs_controller_and_worker_contract(monkeypatch):
    monkeypatch.setitem(
        analysis_request_validator.exareme3_preprocessing_step_classes,
        "test_projection",
        TwoColumnProjection,
    )
    params = {"new_column_names": ["pca_1", "pca_2"]}
    request = _request(
        preprocessing=[
            AnalysisPreprocessingStepDTO(
                name="test_projection",
                parameters=params,
            )
        ]
    )
    inputdata, cdes = analysis_request_validator._validate_and_apply_preprocessing(
        request,
        {"test_projection": _spec()},
        _cdes(),
    )
    transformed, metadata = TwoColumnProjection(
        params=params
    ).transform_data_and_metadata(
        data=pd.DataFrame({"age": [20, 40]}),
        metadata={"age": _cdes()["age"].model_dump()},
    )
    assert inputdata.variables[-2:] == ["pca_1", "pca_2"]
    assert transformed[["pca_1", "pca_2"]].values.tolist() == [[10, 40], [20, 80]]
    for name in params["new_column_names"]:
        assert cdes[name].sql_type == metadata[name]["sql_type"] == "real"
        assert not cdes[name].is_categorical
        assert cdes[name].enumerations is None
    serialized = _convert_transformer_specification_to_dto(_spec()).model_dump(
        mode="json"
    )
    assert serialized["outputs"]["columns"] == [
        {
            "type": "real",
            "stattype": "numerical",
            "column_count_param": None,
            "enumerations": None,
        },
        {
            "type": "real",
            "stattype": "numerical",
            "column_count_param": None,
            "enumerations": None,
        },
    ]


def test_sequential_steps_cannot_reuse_new_column_names(monkeypatch):
    monkeypatch.setitem(
        analysis_request_validator.exareme3_preprocessing_step_classes,
        "test_projection",
        TwoColumnProjection,
    )
    step = AnalysisPreprocessingStepDTO(
        name="test_projection",
        parameters={"new_column_names": ["pca_1", "pca_2"]},
    )
    with pytest.raises(BadUserInput, match="already exists"):
        analysis_request_validator._validate_and_apply_preprocessing(
            _request(preprocessing=[step, step]),
            {"test_projection": _spec()},
            _cdes(),
        )


def test_worker_rejects_output_order_and_metadata_mismatches(monkeypatch):
    creator = TwoColumnProjection(params={"new_column_names": ["pca_1", "pca_2"]})
    transform = creator.transform_data
    monkeypatch.setattr(
        creator,
        "transform_data",
        lambda **kwargs: transform(**kwargs)[["age", "pca_2", "pca_1"]],
    )
    with pytest.raises(ValueError, match="order"):
        creator.transform_data_and_metadata(
            data=pd.DataFrame({"age": [20]}), metadata={}
        )
    monkeypatch.setattr(creator, "transform_data", transform)
    monkeypatch.setattr(creator, "transform_metadata", lambda **kwargs: {})
    with pytest.raises(ValueError, match="Metadata"):
        creator.transform_data_and_metadata(
            data=pd.DataFrame({"age": [20]}), metadata={}
        )


@pytest.mark.parametrize(
    "names",
    [
        None,
        "a",
        [],
        ["a"],
        ["a", "b", "c"],
        ["", "b"],
        [" ", "b"],
        [1, "b"],
        ["a", "a"],
        ["age", "b"],
    ],
)
def test_rejects_invalid_names(names):
    with pytest.raises(BadUserInput):
        resolve_output_metadata(_spec(), {"new_column_names": names}, {"age": {}})


@pytest.mark.parametrize(
    "column",
    [
        {"type": "text", "stattype": "nominal"},
        {"type": "text", "stattype": "numerical"},
        {"type": "real", "stattype": "nominal"},
        {"type": "jsonObject", "stattype": "numerical"},
        {
            "type": "real",
            "stattype": "numerical",
            "enumerations": {
                "source_parameter_name": "count",
                "source_type": "prefixed_range",
                "prefix_parameter_name": "prefix",
            },
        },
    ],
)
def test_rejects_invalid_column_contract(column):
    with pytest.raises(ValidationError):
        specs.PreprocessingOutputColumnSpecification.model_validate(column)


@pytest.mark.parametrize(
    "enums",
    [
        {"source_type": "dict_keys"},
        {
            "source_parameter_name": "rules",
            "source_type": "dict_keys",
            "path": ["nested"],
        },
        {
            "source_parameter_name": "rules",
            "source_type": "dict_keys",
            "prefix_parameter_name": "prefix",
        },
        {"source_parameter_name": "count", "source_type": "prefixed_range"},
        {
            "source_parameter_name": "count",
            "source_type": "prefixed_range",
            "prefix_parameter_name": "prefix",
            "additional_value_parameter_name": "default",
        },
    ],
)
def test_rejects_invalid_enumeration_contract(enums):
    with pytest.raises(ValidationError):
        specs.PreprocessingOutputEnumerationsSpecification.model_validate(enums)


def test_rejects_empty_columns_and_invalid_parameter_references():
    with pytest.raises(ValidationError):
        _spec([])
    for parameter in [
        _parameter(specs.ParameterType.TEXT),
        _parameter(specs.ParameterType.INT, multiple=True),
        _parameter(specs.ParameterType.TEXT, multiple=True, required=False),
    ]:
        with pytest.raises(ValidationError):
            _spec(new_column_names=parameter)
    with pytest.raises(ValidationError):
        _spec(
            [
                {
                    "type": "text",
                    "stattype": "nominal",
                    "enumerations": {
                        "source_parameter_name": "missing",
                        "source_type": "dict_keys",
                    },
                }
            ]
        )
    specification = _spec().model_dump()
    specification["output"] = {}
    with pytest.raises(ValidationError):
        specs.PreprocessingStepSpecification.model_validate(specification)


@pytest.mark.parametrize(
    "name,parameter",
    [
        ("count", _parameter(specs.ParameterType.REAL)),
        ("count", _parameter(specs.ParameterType.INT, required=False)),
        ("prefix", _parameter(specs.ParameterType.TEXT, multiple=True)),
        ("prefix", _parameter(specs.ParameterType.TEXT, required=False)),
    ],
)
def test_rejects_invalid_range_parameter_references(name, parameter):
    specification = _range_spec().model_dump()
    specification["parameters"][name] = parameter.model_dump()
    with pytest.raises(ValidationError):
        specs.PreprocessingStepSpecification.model_validate(specification)


def test_accepts_integer_numerical_and_nominal_columns():
    specification = _spec(
        [
            {"type": "int", "stattype": "numerical"},
            {
                "type": "int",
                "stattype": "nominal",
                "enumerations": {
                    "source_parameter_name": "rules",
                    "source_type": "dict_keys",
                },
            },
        ],
        rules=_parameter(specs.ParameterType.DICT),
    )
    resolved = resolve_output_metadata(
        specification,
        {
            "new_column_names": ["count", "category"],
            "rules": {"1": {}, "2": {}},
        },
        {},
    )
    assert resolved["count"]["sql_type"] == resolved["category"]["sql_type"] == "int"
    assert not resolved["count"]["is_categorical"]
    assert resolved["category"]["enumerations"] == {"1": "1", "2": "2"}


@pytest.mark.parametrize(
    "count", [0, -1, True, 1.5, "2", None, float("nan"), float("inf"), -float("inf")]
)
def test_rejects_invalid_range_count(count):
    with pytest.raises(BadUserInput):
        resolve_output_metadata(
            _range_spec(),
            {
                "new_column_names": ["group"],
                "count": count,
                "prefix": "group_",
            },
            {},
        )


def test_independent_categories_and_mixed_outputs():
    specification = _spec(
        [
            {
                "type": "text",
                "stattype": "nominal",
                "enumerations": {
                    "source_parameter_name": "rules",
                    "source_type": "dict_keys",
                    "additional_value_parameter_name": "default",
                },
            },
            {
                "type": "text",
                "stattype": "nominal",
                "enumerations": {
                    "source_parameter_name": "other_rules",
                    "source_type": "dict_keys",
                },
            },
            {"type": "real", "stattype": "numerical"},
        ],
        rules=_parameter(specs.ParameterType.DICT),
        other_rules=_parameter(specs.ParameterType.DICT),
        default=_parameter(specs.ParameterType.TEXT, required=False),
    )
    params = {
        "new_column_names": ["risk", "response", "score"],
        "rules": {"low": {}},
        "other_rules": {"yes": {}, "no": {}},
        "default": "low",
    }
    resolved = resolve_output_metadata(specification, params, {})
    assert resolved["risk"]["enumerations"] == {"low": "low"}
    assert resolved["response"]["enumerations"] == {"yes": "yes", "no": "no"}
    assert "enumerations" not in resolved["score"]
    params.pop("default")
    assert resolve_output_metadata(specification, params, {}) == resolved


@pytest.mark.parametrize("rules", [{}, [], {"": {}}, {1: {}}, {" ": {}}])
def test_rejects_invalid_dictionary_enumerations(rules):
    with pytest.raises(BadUserInput):
        resolve_output_metadata(
            CategoricalColumnCreator.get_specification(),
            {
                "new_column_names": ["risk"],
                "rules": rules,
            },
            {},
        )


def test_range_generation_and_metadata_contradictions():
    resolved = resolve_output_metadata(
        _range_spec(),
        {
            "new_column_names": ["group"],
            "count": 12,
            "prefix": "custom__9_",
        },
        {},
    )
    assert list(resolved["group"]["enumerations"]) == [
        f"custom__9_{i}" for i in range(1, 13)
    ]
    with pytest.raises(ValueError, match="metadata|Metadata"):
        validate_output_metadata(resolved, {"group": {}}, ["group"])
    with pytest.raises(ValueError, match="order"):
        validate_output_metadata(resolved, resolved, [])


def test_categorical_creator_transform_keeps_default():
    creator = CategoricalColumnCreator(
        params={
            "new_column_names": ["risk"],
            "strategy": "filter_rules",
            "rules": {"high": {"id": "age", "operator": "greater", "value": 65}},
            "default_enumeration": "low",
        }
    )
    data, metadata = creator.transform_data_and_metadata(
        data=pd.DataFrame({"age": [20, 80]}),
        metadata={},
    )
    assert data["risk"].tolist() == ["low", "high"]
    assert metadata["risk"]["enumerations"] == {"high": "high", "low": "low"}
