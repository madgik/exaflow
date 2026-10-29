import asyncio

import pytest
from pydantic import ValidationError
from quart import Quart
from quart import request

import exaflow.controller.services.api.analysis_request_validator as analysis_request_validator
from exaflow.algorithms.exareme3.cluster.kmeans_cluster_creator import (
    KMeansClusterCreator,
)
from exaflow.algorithms.exareme3.feature_engineering.categorical_column_creator import (
    CategoricalColumnCreator,
)
from exaflow.algorithms.exareme3.longitudinal.longitudinal_transformer import (
    LongitudinalTransformer,
)
from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterDictValueType
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import ParameterType
from exaflow.controller.quart.error_handlers import error_handlers
from exaflow.controller.services.api.analysis_request_dtos import AnalysisAlgorithmDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisInputDataDTO
from exaflow.controller.services.api.analysis_request_dtos import (
    AnalysisPreprocessingStepDTO,
)
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.errors import GlobalWorkerUnavailableError
from exaflow.data_filters import FilterError
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement

DATA_MODEL = "dementia:0.1"
_AVAILABLE_GLOBAL_WORKER = object()


class FakeWorkerLandscapeAggregator:
    def __init__(self, global_worker=_AVAILABLE_GLOBAL_WORKER):
        self._global_worker = global_worker

    def get_global_worker(self):
        return self._global_worker

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


def _algorithm_spec(parameters=None):
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
        parameters=parameters,
    )


def _request(*, preprocessing=None, x=None, y=None, variables=None, parameters=None):
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
            parameters={} if parameters is None else parameters,
        ),
        flags={},
    )


def _default_preprocessing_specs():
    return {
        "categorical_column_creator": CategoricalColumnCreator.get_specification(),
        "longitudinal_transformer": LongitudinalTransformer.get_specification(),
    }


def _validate(request, *, algorithms_specs=None, preprocessing_steps_specs=None):
    return analysis_request_validator.validate_analysis_request(
        analysis_request_dto=request,
        algorithms_specs=algorithms_specs or {"sample_algorithm": _algorithm_spec()},
        preprocessing_steps_specs=preprocessing_steps_specs
        or _default_preprocessing_specs(),
        worker_landscape_aggregator=FakeWorkerLandscapeAggregator(),
        smpc_enabled=False,
        smpc_optional=False,
    )


def _validate_parameter_request(parameters, parameter_specs):
    return _validate(
        _request(parameters=parameters),
        algorithms_specs={
            "sample_algorithm": _algorithm_spec(parameters=parameter_specs)
        },
    )


def _post_parameter_request(parameters, parameter_specs):
    app = Quart(__name__)
    app.register_blueprint(error_handlers)
    algorithms_specs = {"sample_algorithm": _algorithm_spec(parameters=parameter_specs)}
    body = _request(parameters=parameters).model_dump(mode="json")

    @app.post("/analysis")
    async def run_analysis():
        analysis_request_dto = AnalysisRequestDTO.model_validate(await request.json)
        analysis_request_validator.validate_analysis_request(
            analysis_request_dto=analysis_request_dto,
            algorithms_specs=algorithms_specs,
            preprocessing_steps_specs=_default_preprocessing_specs(),
            worker_landscape_aggregator=FakeWorkerLandscapeAggregator(),
            smpc_enabled=False,
            smpc_optional=False,
        )
        return "ok"

    async def _call():
        async with app.test_client() as client:
            response = await client.post("/analysis", json=body)
            body_text = await response.get_data(as_text=True)
            return response.status_code, body_text

    return asyncio.run(_call())


def _risk_group_step(rules=None):
    return AnalysisPreprocessingStepDTO(
        name="categorical_column_creator",
        parameters={
            "new_column_names": ["risk_group"],
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


def _int_parameter_spec(**overrides):
    fields = dict(
        label="K",
        desc="k",
        types=[ParameterType.INT],
        required=True,
        multiple=False,
        min=1,
        max=10,
    )
    fields.update(overrides)
    return {"k": ParameterSpecification(**fields)}


def _validate_parameters(value, spec):
    analysis_request_validator._validate_parameters(
        {"k": value},
        spec,
        _request().inputdata,
        _cdes(),
    )


@pytest.mark.parametrize("value", [0, 0.0, -1, True, 11])
def test_parameter_validation_rejects_out_of_range_and_bool_values(value):
    with pytest.raises(BadUserInput):
        _validate_parameters(value, _int_parameter_spec())


def test_parameter_validation_accepts_bool_for_boolean_parameters():
    spec = _int_parameter_spec(types=[ParameterType.BOOLEAN], min=None, max=None)

    _validate_parameters(False, spec)


def _dict_int_parameter_spec():
    return {
        "folds": ParameterSpecification(
            label="Folds",
            desc="Fold for each variable.",
            types=[ParameterType.DICT],
            required=True,
            multiple=False,
            dict_values_type=ParameterDictValueType.INT,
        )
    }


def _message(parameters, parameter_specs):
    with pytest.raises(BadUserInput) as exc_info:
        _validate_parameter_request(parameters, parameter_specs)
    return str(exc_info.value)


def test_out_of_range_int_returns_http_460_with_at_least_message():
    status_code, body = _post_parameter_request({"k": 0}, _int_parameter_spec())

    assert status_code == 460
    assert body == "Parameter 'K' values should be at least 1 ."
    assert "1.0" not in body
    assert "greater than" not in body


@pytest.mark.parametrize("value", [True, False])
def test_real_parameter_rejects_booleans(value):
    spec = _int_parameter_spec(types=[ParameterType.REAL], min=None, max=None)

    message = _message({"k": value}, spec)

    assert "Parameter 'K' values should be of types: ['real']." == message


def test_multi_value_int_rejects_boolean_item():
    spec = _int_parameter_spec(multiple=True, min=None, max=None)

    message = _message({"k": [1, True]}, spec)

    assert "Parameter 'K' values should be of types: ['int']." == message


def test_dict_of_ints_rejects_boolean_and_accepts_zero():
    spec = _dict_int_parameter_spec()

    message = _message({"folds": {"a": True}}, spec)

    assert message == ("Parameter 'Folds' dictionary values should be of type: int.")
    _validate_parameter_request({"folds": {"a": 0}}, spec)


def test_boolean_parameter_rejects_zero_and_accepts_true():
    spec = _int_parameter_spec(types=[ParameterType.BOOLEAN], min=None, max=None)

    message = _message({"k": 0}, spec)

    assert "Parameter 'K' values should be of types: ['boolean']." == message
    _validate_parameter_request({"k": True}, spec)


def test_int_parameter_with_minimum_zero_accepts_zero():
    _validate_parameter_request({"k": 0}, _int_parameter_spec(min=0))


def test_optional_int_rejects_zero_below_minimum_and_accepts_null():
    spec = _int_parameter_spec(required=False)

    message = _message({"k": 0}, spec)

    assert message == "Parameter 'K' values should be at least 1 ."
    _validate_parameter_request({"k": None}, spec)


def test_int_limit_messages_use_at_least_and_at_most_without_decimal():
    below = _message({"k": 0}, _int_parameter_spec())
    above = _message({"k": 11}, _int_parameter_spec())

    assert below == "Parameter 'K' values should be at least 1 ."
    assert above == "Parameter 'K' values should be at most 10 ."


def test_real_limit_messages_keep_decimal_bounds():
    spec = _int_parameter_spec(types=[ParameterType.REAL], min=0.1, max=1.0)
    below = _message({"k": 0.0}, spec)
    above = _message({"k": 1.5}, spec)

    assert below == "Parameter 'K' values should be at least 0.1 ."
    assert above == "Parameter 'K' values should be at most 1.0 ."


@pytest.mark.parametrize(
    "parameters,parameter_specs",
    [
        ({"seed": 1}, None),
        ({"k": "  "}, "int"),
        ({"k": []}, "multiple"),
        ({"k": None}, "int"),
        ({"folds": {}}, "dict"),
    ],
)
def test_blank_parameter_message_uses_label(parameters, parameter_specs):
    if parameter_specs == "multiple":
        specs = _int_parameter_spec(multiple=True, min=None, max=None)
    elif parameter_specs == "dict":
        specs = _dict_int_parameter_spec()
    elif parameter_specs is None:
        specs = {
            "k": _int_parameter_spec()["k"],
            "seed": ParameterSpecification(
                label="Seed",
                desc="Seed.",
                types=[ParameterType.INT],
                required=True,
                multiple=False,
                min=0,
            ),
        }
    else:
        specs = _int_parameter_spec()

    message = _message(parameters, specs)

    assert "should not be blank" in message
    assert "Parameter 'k'" not in message
    assert "Parameter 'folds'" not in message
    assert message.startswith("Parameter 'K'") or message.startswith(
        "Parameter 'Folds'"
    )


def test_preprocessing_parameter_rejects_zero_below_minimum():
    request = _request(
        preprocessing=[
            AnalysisPreprocessingStepDTO(
                name="kmeans_cluster_creator",
                parameters={"new_column_names": ["cluster"], "k": 0},
            )
        ]
    )
    specs = _default_preprocessing_specs()
    specs["kmeans_cluster_creator"] = KMeansClusterCreator.get_specification()

    with pytest.raises(BadUserInput) as exc_info:
        _validate(request, preprocessing_steps_specs=specs)

    assert (
        str(exc_info.value)
        == "Parameter 'Number of clusters' values should be at least 1 ."
    )


def test_validation_datasets_without_global_worker_report_it_unavailable():
    request = _request()
    request = request.model_copy(
        update={
            "inputdata": request.inputdata.model_copy(
                update={"validation_datasets": ["validation_a"]}
            )
        }
    )

    with pytest.raises(GlobalWorkerUnavailableError):
        analysis_request_validator.validate_analysis_request(
            analysis_request_dto=request,
            algorithms_specs={"sample_algorithm": _algorithm_spec()},
            preprocessing_steps_specs=_default_preprocessing_specs(),
            worker_landscape_aggregator=FakeWorkerLandscapeAggregator(
                global_worker=None
            ),
            smpc_enabled=False,
            smpc_optional=False,
        )


def test_training_only_request_runs_without_global_worker():
    analysis_request_validator.validate_analysis_request(
        analysis_request_dto=_request(),
        algorithms_specs={"sample_algorithm": _algorithm_spec()},
        preprocessing_steps_specs=_default_preprocessing_specs(),
        worker_landscape_aggregator=FakeWorkerLandscapeAggregator(global_worker=None),
        smpc_enabled=False,
        smpc_optional=False,
    )
