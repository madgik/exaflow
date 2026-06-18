import pytest
from pydantic import ValidationError

import exaflow.controller.services.api.analysis_request_validator as analysis_request_validator
from exaflow.algorithms.exareme3.preprocessing.categorical_column_creator import (
    CategoricalColumnCreator,
)
from exaflow.algorithms.exareme3.preprocessing.longitudinal_transformer import (
    LongitudinalTransformer,
)
from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterDictValueType
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import ParameterType
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
