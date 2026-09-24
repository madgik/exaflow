import pytest
from pydantic import ValidationError

import exaflow.controller.services.api.analysis_request_validator as analysis_request_validator
from exaflow.algorithms.exareme3.linear_model.linear_regression import LinearRegression
from exaflow.algorithms.exareme3.preprocessing.categorical_column_creator import (
    CategoricalColumnCreator,
)
from exaflow.algorithms.exareme3.preprocessing.longitudinal_transformer import (
    LongitudinalTransformer,
)
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterDictValueType
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import ParameterType
from exaflow.algorithms.specifications import PreprocessingOutputSpecification
from exaflow.algorithms.specifications import PreprocessingOutputType
from exaflow.algorithms.specifications import PreprocessingStepSpecification
from exaflow.controller.services.api.analysis_request_dtos import AnalysisAlgorithmDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisInputDataDTO
from exaflow.controller.services.api.analysis_request_dtos import (
    AnalysisPreprocessingStepDTO,
)
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.data_filters import FilterError
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement

DATA_MODEL = "dementia:0.1"


class FakeWorkerLandscapeAggregator:
    def get_training_and_validation_datasets(self, data_model):
        return ["dataset_a"], ["validation_a"]

    def get_cdes(self, data_model):
        return _cdes()


def _cde(code, sql_type, *, categorical=False, enumerations=None):
    return CommonDataElement(
        code=code,
        label=code,
        sql_type=sql_type,
        is_categorical=categorical,
        enumerations=enumerations,
    )


def _cdes():
    return {
        "age": _cde("age", "int"),
        "gender": _cde(
            "gender",
            "text",
            categorical=True,
            enumerations={"M": "M", "F": "F"},
        ),
        "diagnosis": _cde(
            "diagnosis",
            "text",
            categorical=True,
            enumerations={"AD": "AD", "CN": "CN"},
        ),
        "outcome": _cde(
            "outcome",
            "text",
            categorical=True,
            enumerations={"yes": "yes", "no": "no"},
        ),
        "visitid": _cde(
            "visitid",
            "text",
            categorical=True,
            enumerations={"BL": "BL", "FL1": "FL1"},
        ),
    }


def _algorithm_spec():
    return AlgorithmSpecification(
        name="sample_algorithm",
        desc="sample",
        documentation="sample",
        label="Sample Algorithm",
        enabled=True,
        y=InputDataSpecification(
            label="Outcome",
            desc="Outcome variable.",
            types=[InputDataType.TEXT],
            stattypes=[InputDataStatType.NOMINAL],
            required=True,
            min_count=1,
            max_count=1,
        ),
        x=InputDataSpecification(
            label="Features",
            desc="Feature variables.",
            types=[InputDataType.TEXT],
            stattypes=[InputDataStatType.NOMINAL],
            required=True,
            min_count=1,
        ),
    )


def _request(*, preprocessing=None, x=None, y=None, variables=None):
    return AnalysisRequestDTO(
        inputdata=AnalysisInputDataDTO(
            data_model=DATA_MODEL,
            datasets=["dataset_a"],
            variables=variables or ["age", "gender", "diagnosis", "outcome"],
        ),
        preprocessing=preprocessing,
        algorithm=AnalysisAlgorithmDTO(
            name="sample_algorithm",
            x=x or ["gender"],
            y=y or ["outcome"],
            parameters={},
        ),
        flags={},
    )


def _validate(request):
    return analysis_request_validator.validate_analysis_request(
        analysis_request_dto=request,
        algorithms_specs={"sample_algorithm": _algorithm_spec()},
        preprocessing_steps_specs={
            "categorical_column_creator": CategoricalColumnCreator.get_specification(),
            "longitudinal_transformer": LongitudinalTransformer.get_specification(),
        },
        worker_landscape_aggregator=FakeWorkerLandscapeAggregator(),
        smpc_enabled=False,
        smpc_optional=False,
    )


def _risk_group_step(rules=None):
    return AnalysisPreprocessingStepDTO(
        name="categorical_column_creator",
        parameters={
            "code": "risk_group",
            "strategy": "filter_rules",
            "rules": rules
            or {
                "high": {
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "age",
                            "operator": "greater_or_equal",
                            "value": 80,
                        }
                    ],
                },
                "medium": {
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "diagnosis",
                            "operator": "equal",
                            "value": "AD",
                        }
                    ],
                },
            },
            "default_enumeration": "low",
        },
    )


def test_valid_analysis_request_uses_inputdata_variables_and_algorithm_xy():
    _validate(_request())


def test_old_inputdata_xy_shape_fails_dto_validation():
    with pytest.raises(ValidationError):
        AnalysisRequestDTO.model_validate(
            {
                "inputdata": {
                    "data_model": DATA_MODEL,
                    "datasets": ["dataset_a"],
                    "x": ["gender"],
                    "y": ["outcome"],
                    "variables": ["gender", "outcome"],
                },
                "algorithm": {
                    "name": "sample_algorithm",
                    "x": ["gender"],
                    "y": ["outcome"],
                },
            }
        )


def test_top_level_parameters_shape_fails_dto_validation():
    with pytest.raises(ValidationError):
        AnalysisRequestDTO.model_validate(
            {
                "inputdata": {
                    "data_model": DATA_MODEL,
                    "datasets": ["dataset_a"],
                    "variables": ["gender", "outcome"],
                },
                "parameters": {},
                "algorithm": {
                    "name": "sample_algorithm",
                    "x": ["gender"],
                    "y": ["outcome"],
                },
            }
        )


def test_dict_preprocessing_shape_fails_dto_validation():
    with pytest.raises(ValidationError):
        AnalysisRequestDTO.model_validate(
            {
                "inputdata": {
                    "data_model": DATA_MODEL,
                    "datasets": ["dataset_a"],
                    "variables": ["gender", "outcome"],
                },
                "preprocessing": {"categorical_column_creator": {}},
                "algorithm": {
                    "name": "sample_algorithm",
                    "x": ["gender"],
                    "y": ["outcome"],
                },
            }
        )


def test_dict_parameter_with_filter_values_is_validated():
    parameter = ParameterSpecification(
        label="Enumeration filters",
        desc="Dictionary where each value is a filter.",
        types=[ParameterType.DICT],
        required=True,
        multiple=False,
        dict_values_type=ParameterDictValueType.FILTER,
    )

    analysis_request_validator._validate_parameters(
        parameters={
            "rules": {
                "high": {
                    "condition": "AND",
                    "rules": [{"id": "age", "operator": "greater", "value": 70}],
                }
            }
        },
        parameters_specs={"rules": parameter},
        inputdata=analysis_request_validator._build_source_inputdata(_request()),
        data_model_cdes=_cdes(),
    )


def test_rejects_rules_dict_value_that_is_not_a_filter():
    with pytest.raises(FilterError, match="Filter type can only be dict"):
        _validate(_request(preprocessing=[_risk_group_step(rules={"high": "bad"})]))


def test_derived_categorical_cde_is_available_to_algorithm_x():
    _validate(_request(preprocessing=[_risk_group_step()], x=["risk_group", "gender"]))


def _numerical_output_spec():
    return PreprocessingStepSpecification(
        name="numerical_creator",
        desc="Creates one numerical column.",
        documentation="Creates one numerical column.",
        label="Numerical creator",
        enabled=True,
        parameters={},
        output=PreprocessingOutputSpecification(
            type=PreprocessingOutputType.NEW_NUMERICAL_COLUMN,
            code_parameter="code",
        ),
    )


class FakeMultipleNumericalCreator(PreprocessingStep):
    def __init__(self, *, params):
        super().__init__(params=params)
        self._codes = list(params["codes"])

    @classmethod
    def get_specification(cls):
        return _multiple_numerical_output_spec()

    def validate_params(self, *, inputdata, metadata):
        return None

    def transform_variables(self, *, variables):
        return list(variables) + self._codes

    def transform_metadata(self, *, metadata):
        return dict(metadata)

    def transform_data(self, *, data):
        return data


def _multiple_numerical_output_spec():
    return PreprocessingStepSpecification(
        name="fake_multiple_numerical_creator",
        desc="Creates numerical columns.",
        documentation="Creates numerical columns.",
        label="Multiple numerical creator",
        enabled=True,
        parameters={
            "codes": ParameterSpecification(
                label="Generated codes",
                desc="Generated numerical column codes.",
                types=[ParameterType.TEXT],
                required=True,
                multiple=True,
            )
        },
        output=PreprocessingOutputSpecification(
            type=PreprocessingOutputType.NEW_NUMERICAL_COLUMN,
            multiple=True,
        ),
    )


def _validate_with_fake_multiple_creator(monkeypatch, codes, *, variables=None):
    monkeypatch.setitem(
        analysis_request_validator.exareme3_preprocessing_step_classes,
        "fake_multiple_numerical_creator",
        FakeMultipleNumericalCreator,
    )
    request = _request(
        variables=variables or ["age", "gender", "diagnosis", "outcome"],
        preprocessing=[
            AnalysisPreprocessingStepDTO(
                name="fake_multiple_numerical_creator",
                parameters={"codes": codes},
            )
        ],
    )
    return analysis_request_validator._validate_and_apply_preprocessing(
        analysis_request_dto=request,
        preprocessing_steps_specs={
            "fake_multiple_numerical_creator": _multiple_numerical_output_spec()
        },
        data_model_cdes=_cdes(),
    )


def test_multiple_numerical_outputs_preserve_order_and_metadata(monkeypatch):
    transformed_inputdata, transformed_cdes = _validate_with_fake_multiple_creator(
        monkeypatch,
        ["derived_b", "derived_a"],
    )

    assert transformed_inputdata.variables[-2:] == ["derived_b", "derived_a"]
    assert transformed_cdes["derived_b"].model_dump() == {
        "code": "derived_b",
        "label": "derived_b",
        "sql_type": "real",
        "is_categorical": False,
        "enumerations": None,
        "min": None,
        "max": None,
    }
    assert transformed_cdes["derived_a"].model_dump() == {
        "code": "derived_a",
        "label": "derived_a",
        "sql_type": "real",
        "is_categorical": False,
        "enumerations": None,
        "min": None,
        "max": None,
    }


def test_multiple_numerical_collision_rejects_whole_step(monkeypatch):
    with pytest.raises(BadUserInput, match="cannot create CDE 'age'"):
        _validate_with_fake_multiple_creator(
            monkeypatch,
            ["derived_a", "age"],
        )


def test_multiple_numerical_duplicate_output_rejects_whole_step(monkeypatch):
    with pytest.raises(BadUserInput, match="duplicate variables"):
        _validate_with_fake_multiple_creator(
            monkeypatch,
            ["derived_a", "derived_a"],
        )


def test_multiple_numerical_blank_output_rejects_whole_step(monkeypatch):
    with pytest.raises(BadUserInput, match="blank variable name"):
        _validate_with_fake_multiple_creator(
            monkeypatch,
            ["derived_a", "   "],
        )


def test_multiple_numerical_failure_does_not_commit_variables_or_metadata(monkeypatch):
    original_variables = ["age", "gender", "diagnosis", "outcome"]
    with pytest.raises(BadUserInput):
        _validate_with_fake_multiple_creator(
            monkeypatch,
            ["derived_a", "age"],
            variables=original_variables,
        )

    assert original_variables == ["age", "gender", "diagnosis", "outcome"]
    assert "derived_a" not in _cdes()


def test_downstream_validation_sees_all_multiple_numerical_outputs(monkeypatch):
    _, transformed_cdes = _validate_with_fake_multiple_creator(
        monkeypatch,
        ["derived_b", "derived_a"],
    )
    transformed_cdes["derived_b"] = transformed_cdes["derived_b"]
    transformed_cdes["derived_a"] = transformed_cdes["derived_a"]

    analysis_request_validator._validate_algorithm_inputdatas(
        x=["derived_b", "derived_a"],
        y=["age"],
        algorithm_specs=LinearRegression.get_specification(),
        data_model_cdes={**_cdes(), **transformed_cdes},
    )


def test_numerical_output_metadata_is_derived_for_one_generated_column():
    metadata = {name: cde.model_dump() for name, cde in _cdes().items()}
    spec = _numerical_output_spec()

    analysis_request_validator._validate_preprocessing_output_name(
        preprocessing_step_spec=spec,
        params={"code": "derived_x"},
        data_model_cdes=metadata,
    )
    analysis_request_validator._derive_preprocessing_output_metadata(
        preprocessing_step_spec=spec,
        params={"code": "derived_x"},
        metadata=metadata,
    )

    assert metadata["derived_x"] == {
        "code": "derived_x",
        "label": "derived_x",
        "sql_type": "real",
        "is_categorical": False,
        "enumerations": None,
    }


def test_numerical_output_collision_is_rejected():
    with pytest.raises(BadUserInput, match="cannot create CDE 'age'"):
        analysis_request_validator._validate_preprocessing_output_name(
            preprocessing_step_spec=_numerical_output_spec(),
            params={"code": "age"},
            data_model_cdes=_cdes(),
        )


def test_numerical_output_blank_code_is_rejected():
    with pytest.raises(BadUserInput, match="non-blank generated code"):
        analysis_request_validator._validate_preprocessing_output_name(
            preprocessing_step_spec=_numerical_output_spec(),
            params={"code": "   "},
            data_model_cdes=_cdes(),
        )


def test_downstream_validation_accepts_a_generated_numerical_cde():
    data_model_cdes = _cdes()
    data_model_cdes["derived_x"] = _cde("derived_x", "real")

    analysis_request_validator._validate_algorithm_inputdatas(
        x=["derived_x"],
        y=["age"],
        algorithm_specs=LinearRegression.get_specification(),
        data_model_cdes=data_model_cdes,
    )


def test_derived_categorical_cde_contains_rule_and_default_enumerations():
    transformed_inputdata, transformed_cdes = (
        analysis_request_validator._validate_and_apply_preprocessing(
            analysis_request_dto=_request(preprocessing=[_risk_group_step()]),
            preprocessing_steps_specs={
                "categorical_column_creator": CategoricalColumnCreator.get_specification()
            },
            data_model_cdes=_cdes(),
        )
    )

    assert "risk_group" in transformed_inputdata.variables
    assert transformed_cdes["risk_group"].model_dump() == {
        "code": "risk_group",
        "label": "risk_group",
        "sql_type": "text",
        "is_categorical": True,
        "enumerations": {"high": "high", "medium": "medium", "low": "low"},
        "min": None,
        "max": None,
    }


def test_rejects_algorithm_x_unknown_after_preprocessing():
    with pytest.raises(BadUserInput, match="does not exist in the data model"):
        _validate(_request(x=["unknown"]))


def test_rejects_preprocessing_filter_referencing_unavailable_variable():
    rules = {
        "high": {
            "condition": "AND",
            "rules": [{"id": "age", "operator": "greater", "value": 70}],
        }
    }

    with pytest.raises(FilterError, match="Column age does not exist"):
        _validate(
            _request(
                preprocessing=[_risk_group_step(rules=rules)],
                variables=["gender", "diagnosis", "outcome"],
            )
        )


def test_preprocessing_filter_rejects_numeric_strings_for_numeric_cdes():
    rules = {
        "high": {
            "condition": "AND",
            "rules": [{"id": "age", "operator": "greater", "value": "70"}],
        }
    }

    with pytest.raises(FilterError, match="age's type: int"):
        _validate(_request(preprocessing=[_risk_group_step(rules=rules)]))


def test_longitudinal_fixed_cdes_are_available_for_parameter_validation():
    request = _request(
        preprocessing=[
            AnalysisPreprocessingStepDTO(
                name="longitudinal_transformer",
                parameters={
                    "visit1": "BL",
                    "visit2": "FL1",
                    "strategies": {
                        "age": "diff",
                        "gender": "first",
                        "outcome": "first",
                    },
                },
            )
        ],
        variables=["age", "gender", "outcome"],
        x=["gender"],
    )

    _validate(request)
