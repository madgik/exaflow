import asyncio

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

import exaflow.controller.services.api.analysis_request_validator as analysis_request_validator
from exaflow import exareme3_algorithm_classes
from exaflow.algorithms.exareme3.cluster.kmeans_cluster_creator import (
    KMeansClusterCreator,
)
from exaflow.algorithms.exareme3.feature_engineering.categorical_column_creator import (
    CategoricalColumnCreator,
)
from exaflow.algorithms.exareme3.longitudinal.longitudinal_transformer import (
    LongitudinalTransformer,
)
from exaflow.algorithms.exareme3.missing_values.missing_values_handler import (
    MissingValuesHandler,
)
from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import ParameterDictValueType
from exaflow.algorithms.specifications import ParameterEnumSpecification
from exaflow.algorithms.specifications import ParameterEnumType
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import ParameterType
from exaflow.controller.http.error_handlers import register_error_handlers
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
from tests.standalone_tests.controller.asgi_client import controller_client

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


def _cde(code, cde_type, *, enumerations=None):
    return CommonDataElement(
        code=code,
        label=code,
        type=cde_type,
        enumerations=enumerations,
    )


def _cdes():
    return {
        "age": _cde("age", "discrete"),
        "gender": _cde(
            "gender",
            "nominal",
            enumerations={"1": "M", "2": "F"},
        ),
        "diagnosis": _cde(
            "diagnosis",
            "nominal",
            enumerations={"1": "AD", "2": "CN"},
        ),
        "outcome": _cde(
            "outcome",
            "nominal",
            enumerations={"1": "yes", "2": "no"},
        ),
        "visitid": _cde(
            "visitid",
            "nominal",
            enumerations={"1": "BL", "2": "FL1"},
        ),
        "stage": _cde("stage", "ordinal", enumerations={"1": "I", "2": "II"}),
        "smoker": _cde("smoker", "binary", enumerations={"0": "n", "1": "y"}),
        "notes": _cde("notes", "nominal"),
        "dataset": _cde("dataset", "nominal", enumerations={"1": "dataset_a"}),
    }


def _algorithm_spec(
    parameters=None,
    *,
    y_types=(InputDataStatType.NOMINAL,),
    x_types=(InputDataStatType.NOMINAL,),
):
    return AlgorithmSpecification(
        name="sample_algorithm",
        desc="sample",
        documentation="sample",
        label="Sample Algorithm",
        enabled=True,
        y=InputDataSpecification(
            label="Outcome",
            desc="Outcome variable.",
            types=list(y_types),
            required=True,
            min_count=1,
            max_count=1,
        ),
        x=InputDataSpecification(
            label="Features",
            desc="Feature variables.",
            types=list(x_types),
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
    app = FastAPI()
    register_error_handlers(app)
    algorithms_specs = {"sample_algorithm": _algorithm_spec(parameters=parameter_specs)}
    body = _request(parameters=parameters).model_dump(mode="json")

    @app.post("/analysis")
    async def run_analysis(analysis_request_dto: AnalysisRequestDTO):
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
        async with controller_client(app) as client:
            response = await client.post("/analysis", json=body)
            body_text = response.text
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
                            "value": 1,
                        }
                    ],
                },
            },
            "default_enumeration": "low",
        },
    )


def test_valid_analysis_request_uses_inputdata_variables_and_algorithm_xy():
    _validate(_request())


def _validate_types(*, x, y, x_types=None, y_types=None):
    spec = _algorithm_spec(
        x_types=x_types or (InputDataStatType.NOMINAL,),
        y_types=y_types or (InputDataStatType.NOMINAL,),
    )
    return _validate(
        _request(x=x, y=y, variables=x + y),
        algorithms_specs={"sample_algorithm": spec},
    )


ANY_CATEGORICAL = (
    InputDataStatType.NOMINAL,
    InputDataStatType.ORDINAL,
    InputDataStatType.BINARY,
)


def test_slot_accepting_all_categorical_types_accepts_each_categorical_cde():
    _validate_types(
        x=["stage", "gender"],
        y=["smoker"],
        x_types=ANY_CATEGORICAL,
        y_types=ANY_CATEGORICAL,
    )


@pytest.mark.parametrize("cde", ["stage", "smoker"])
def test_nominal_slot_rejects_ordinal_and_binary_cdes(cde):
    with pytest.raises(BadUserInput, match=f"'{cde}'.*only \\['nominal'\\]"):
        _validate_types(x=["gender"], y=[cde])


@pytest.mark.parametrize("cde", ["gender", "stage"])
def test_binary_slot_rejects_non_binary_categorical_cdes(cde):
    with pytest.raises(BadUserInput, match=f"'{cde}'.*only \\['binary'\\]"):
        _validate_types(
            x=["age"],
            y=[cde],
            x_types=(InputDataStatType.NUMERICAL,),
            y_types=(InputDataStatType.BINARY,),
        )


def test_binary_slot_accepts_binary_cde():
    _validate_types(
        x=["age"],
        y=["smoker"],
        x_types=(InputDataStatType.NUMERICAL,),
        y_types=(InputDataStatType.BINARY,),
    )


def test_two_level_nominal_cde_is_not_accepted_as_binary():
    # `gender` has exactly two levels but is declared nominal.
    with pytest.raises(BadUserInput, match="'gender'.*has type 'nominal'"):
        _validate_types(
            x=["age"],
            y=["gender"],
            x_types=(InputDataStatType.NUMERICAL,),
            y_types=(InputDataStatType.BINARY,),
        )


@pytest.mark.parametrize("cde", ["gender", "smoker", "diagnosis"])
def test_ordinal_slot_rejects_non_ordinal_categorical_cdes(cde):
    with pytest.raises(BadUserInput, match=f"'{cde}'.*only \\['ordinal'\\]"):
        _validate_types(
            x=["age"],
            y=[cde],
            x_types=(InputDataStatType.NUMERICAL,),
            y_types=(InputDataStatType.ORDINAL,),
        )


def test_ordinal_slot_accepts_ordinal_cde():
    _validate_types(
        x=["age"],
        y=["stage"],
        x_types=(InputDataStatType.NUMERICAL,),
        y_types=(InputDataStatType.ORDINAL,),
    )


@pytest.mark.parametrize("cde", ["stage", "smoker", "gender"])
def test_numerical_slot_rejects_every_categorical_cde(cde):
    with pytest.raises(BadUserInput, match=f"'{cde}'.*should NOT be categorical"):
        _validate_types(
            x=["age"],
            y=[cde],
            x_types=(InputDataStatType.NUMERICAL,),
            y_types=(InputDataStatType.NUMERICAL,),
        )


def test_binary_slot_rejects_numerical_cde():
    with pytest.raises(BadUserInput, match="'age'.*should be categorical"):
        _validate_types(
            x=["stage"],
            y=["age"],
            y_types=(InputDataStatType.BINARY,),
            x_types=ANY_CATEGORICAL,
        )


def _event_var_spec(variable_types):
    return {
        "event_var": ParameterSpecification(
            label="Event variable",
            desc="Event variable.",
            types=[ParameterType.TEXT],
            required=True,
            multiple=False,
            enums=ParameterEnumSpecification(
                type=ParameterEnumType.INPUT_VAR_NAMES, source=["x"]
            ),
            variable_types=variable_types,
        )
    }


def _validate_event_var(event_var, variable_types):
    spec = _algorithm_spec(
        _event_var_spec(variable_types),
        y_types=(InputDataStatType.NUMERICAL,),
        x_types=ANY_CATEGORICAL,
    )
    return _validate(
        _request(
            x=["smoker", "gender"],
            y=["age"],
            variables=["smoker", "gender", "age"],
            parameters={"event_var": event_var},
        ),
        algorithms_specs={"sample_algorithm": spec},
    )


def test_parameter_selected_variable_must_match_binary_stattype():
    _validate_event_var("smoker", [InputDataStatType.BINARY])


def test_parameter_selected_nominal_variable_is_rejected_for_binary_stattype():
    with pytest.raises(
        BadUserInput,
        match="'gender'.*selected by parameter 'Event variable'.*only \\['binary'\\]",
    ):
        _validate_event_var("gender", [InputDataStatType.BINARY])


def test_parameter_selected_variable_without_types_is_not_type_checked():
    _validate_event_var("gender", None)


def test_variable_types_requires_input_var_names_enums():
    with pytest.raises(ValidationError, match="variable_types"):
        ParameterSpecification(
            label="Param",
            desc="Param.",
            types=[ParameterType.TEXT],
            required=True,
            multiple=False,
            variable_types=[InputDataStatType.BINARY],
        )


def test_rejects_categorical_cde_without_enumerations():
    with pytest.raises(BadUserInput, match="without enumerations"):
        _validate(_request(y=["notes"], variables=["gender", "notes"]))


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
        inputdata=analysis_request_validator._build_source_inputdata(
            _request(), _cdes()
        ),
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
        "type": "nominal",
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

    with pytest.raises(FilterError, match="age's type: discrete"):
        _validate(_request(preprocessing=[_risk_group_step(rules=rules)]))


def test_longitudinal_fixed_cdes_are_available_for_parameter_validation():
    request = _request(
        preprocessing=[
            AnalysisPreprocessingStepDTO(
                name="longitudinal_transformer",
                parameters={
                    "visit1": "1",
                    "visit2": "2",
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


def _real_algorithm_request(name, *, x, y, parameters):
    return AnalysisRequestDTO(
        inputdata=AnalysisInputDataDTO(
            data_model=DATA_MODEL,
            datasets=["dataset_a"],
            variables=[*x, *y],
        ),
        preprocessing=[
            AnalysisPreprocessingStepDTO(
                name="missing_values_handler",
                parameters={"strategies": {var: "drop" for var in [*x, *y]}},
            )
        ],
        algorithm=AnalysisAlgorithmDTO(name=name, x=x, y=y, parameters=parameters),
        flags={},
    )


def _validate_real_algorithm(name, **kwargs):
    algorithm_class = exareme3_algorithm_classes[name]
    return _validate(
        _real_algorithm_request(name, **kwargs),
        algorithms_specs={name: algorithm_class.get_specification()},
        preprocessing_steps_specs={
            "missing_values_handler": MissingValuesHandler.get_specification()
        },
    )


def test_logistic_regression_accepts_a_binary_outcome():
    _validate_real_algorithm(
        "logistic_regression",
        x=["age"],
        y=["smoker"],
        parameters={"positive_class": "1"},
    )


def test_logistic_regression_accepts_an_integer_positive_class():
    _validate_real_algorithm(
        "logistic_regression",
        x=["age"],
        y=["smoker"],
        parameters={"positive_class": 1},
    )


def test_logistic_regression_rejects_an_undeclared_integer_positive_class():
    with pytest.raises(BadUserInput, match="Positive class"):
        _validate_real_algorithm(
            "logistic_regression",
            x=["age"],
            y=["smoker"],
            parameters={"positive_class": 7},
        )


@pytest.mark.parametrize("outcome", ["outcome", "gender", "stage"])
def test_logistic_regression_rejects_a_non_binary_outcome(outcome):
    with pytest.raises(BadUserInput, match=f"'{outcome}'.*only \\['binary'\\]"):
        _validate_real_algorithm(
            "logistic_regression",
            x=["age"],
            y=[outcome],
            parameters={"positive_class": "yes"},
        )


def test_glmm_ordinal_accepts_an_ordinal_outcome_and_derives_the_order():
    _validate_real_algorithm(
        "glmm_ordinal",
        x=["age", "gender"],
        y=["stage"],
        parameters={"grouping_var": ["gender"]},
    )


@pytest.mark.parametrize("outcome", ["outcome", "smoker"])
def test_glmm_ordinal_rejects_a_non_ordinal_outcome(outcome):
    with pytest.raises(BadUserInput, match=f"'{outcome}'.*only \\['ordinal'\\]"):
        _validate_real_algorithm(
            "glmm_ordinal",
            x=["age", "gender"],
            y=[outcome],
            parameters={"grouping_var": ["gender"], "category_order": ["no", "yes"]},
        )


@pytest.mark.parametrize("event_var", ["gender", "stage"])
def test_cox_regression_rejects_a_non_binary_event_variable(event_var):
    with pytest.raises(BadUserInput, match=f"'{event_var}'.*selected by parameter"):
        _validate_real_algorithm(
            "cox_regression_classical",
            x=[event_var, "smoker"],
            y=["age"],
            parameters={"event_var": event_var, "positive_class": "1"},
        )


def test_cox_regression_accepts_a_binary_event_variable():
    _validate_real_algorithm(
        "cox_regression_classical",
        x=["smoker", "gender"],
        y=["age"],
        parameters={"event_var": "smoker", "positive_class": "1"},
    )


@pytest.mark.parametrize("name", ["cox_regression_classical", "cox_regression_stacked"])
def test_cox_regression_requires_positive_class(name):
    with pytest.raises(BadUserInput, match="Event of interest"):
        _validate_real_algorithm(
            name,
            x=["smoker", "gender"],
            y=["age"],
            parameters={"event_var": "smoker"},
        )


def test_fisher_exact_rejects_a_nominal_variable_even_with_two_levels():
    with pytest.raises(BadUserInput, match="'gender'.*only \\['binary'\\]"):
        _validate_real_algorithm(
            "fisher_exact", x=["gender"], y=["smoker"], parameters={}
        )


def test_source_inputdata_holds_dataset_codes_for_requested_labels():
    inputdata = analysis_request_validator._build_source_inputdata(_request(), _cdes())

    assert inputdata.datasets == ["1"]
