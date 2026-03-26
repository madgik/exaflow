from unittest.mock import patch

import pytest

from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataSpecifications
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterEnumSpecification
from exaflow.algorithms.specifications import ParameterEnumType
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import ParameterType
from exaflow.algorithms.specifications import PreprocessingStep
from exaflow.algorithms.specifications import PreprocessingStepSpecification
from exaflow.controller import DeploymentType
from exaflow.controller import logger as ctrl_logger
from exaflow.controller.services.api.algorithm_request_dtos import AlgorithmInputDataDTO
from exaflow.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exaflow.controller.services.api.algorithm_request_validator import BadRequest
from exaflow.controller.services.api.algorithm_request_validator import (
    validate_algorithm_request,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelRegistry,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsCDES,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DatasetsLocations,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    _wlaRegistries,
)
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement
from exaflow.worker_communication import CommonDataElements


@pytest.fixture
def worker_landscape_aggregator():
    worker_landscape_aggregator = WorkerLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=0,
        tasks_timeout=0,
        deployment_type=DeploymentType.LOCAL,
        localworkers={},
    )

    data_models = {
        "data_model_with_all_cde_types:0.1": CommonDataElements(
            values={
                "int_cde": CommonDataElement(
                    code="int_cde",
                    label="int_cde",
                    sql_type="int",
                    is_categorical=False,
                ),
                "real_cde": CommonDataElement(
                    code="real_cde",
                    label="real_cde",
                    sql_type="real",
                    is_categorical=False,
                ),
                "text_cde_categ": CommonDataElement(
                    code="text_cde_categ",
                    label="text_cde_categ",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"male": "male", "female": "female"},
                ),
                "text_cde_non_categ": CommonDataElement(
                    code="text_cde_non_categ",
                    label="text_cde_non_categ",
                    sql_type="text",
                    is_categorical=False,
                ),
                "int_cde_categ": CommonDataElement(
                    code="int_cde_categ",
                    label="int_cde_categ",
                    sql_type="int",
                    is_categorical=True,
                    enumerations={
                        "1": "1",
                        "2": "2",
                    },
                ),
                "text_cde_3_enums": CommonDataElement(
                    code="text_cde_3_enums",
                    label="text_cde_3_enums",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"male": "male", "female": "female", "Other": "Other"},
                ),
            }
        ),
        "sample_data_model:0.1": CommonDataElements(
            values={
                "sample_cde": CommonDataElement(
                    code="sample_cde",
                    label="sample_cde",
                    sql_type="int",
                    is_categorical=False,
                ),
            }
        ),
    }
    _data_model_registry = DataModelRegistry(
        data_models_cdes=DataModelsCDES(data_models_cdes=data_models),
        datasets_locations=DatasetsLocations(
            datasets_locations={
                "data_model_with_all_cde_types:0.1": {
                    "sample_dataset1": "sample_worker",
                    "sample_dataset2": "sample_worker",
                    "sample_dataset3": "globalworker",
                },
                "sample_data_model:0.1": {"sample_dataset": "sample_worker"},
            }
        ),
    )
    worker_landscape_aggregator._registries = _wlaRegistries(
        data_model_registry=_data_model_registry
    )

    return worker_landscape_aggregator


@pytest.fixture(scope="module")
def algorithms_specs():
    return {
        "algorithm_with_y_int": AlgorithmSpecification(
            name="algorithm_with_y_int",
            desc="algorithm_with_y_int",
            label="algorithm_with_y_int",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                ),
            ),
        ),
        "algorithm_with_y_int_and_validation": AlgorithmSpecification(
            name="algorithm_with_y_int_and_validation",
            desc="algorithm_with_y_int_and_validation",
            label="algorithm_with_y_int_and_validation",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                ),
                validation=True,
            ),
        ),
        "algorithm_with_x_int_and_y_text": AlgorithmSpecification(
            name="algorithm_with_x_int_and_y_text",
            desc="algorithm_with_x_int_and_y_text",
            label="algorithm_with_x_int_and_y_text",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                ),
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                ),
            ),
        ),
        "algorithm_with_y_text_multiple_true": AlgorithmSpecification(
            name="algorithm_with_y_text_multiple_true",
            desc="algorithm_with_y_text_multiple_true",
            label="algorithm_with_y_text_multiple_true",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL, InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=True,
                ),
            ),
        ),
        "algorithm_with_y_text_categ": AlgorithmSpecification(
            name="algorithm_with_y_text_categ",
            desc="algorithm_with_y_text_categ",
            label="algorithm_with_y_text_categ",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=True,
                ),
            ),
        ),
        "algorithm_with_y_text_non_categ": AlgorithmSpecification(
            name="algorithm_with_y_text_non_categ",
            desc="algorithm_with_y_text_non_categ",
            label="algorithm_with_y_text_non_categ",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=True,
                ),
            ),
        ),
        "algorithm_with_variable_enumslen": AlgorithmSpecification(
            name="algorithm_with_variable_enumslen",
            desc="algorithm_with_variable_enumslen",
            label="algorithm_with_variable_enumslen",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=2,
                ),
            ),
        ),
        "algorithm_with_y_and_x_optional": AlgorithmSpecification(
            name="algorithm_with_y_and_x_optional",
            desc="algorithm_with_y_and_x_optional",
            label="algorithm_with_y_and_x_optional",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    required=False,
                    multiple=False,
                ),
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                ),
            ),
        ),
        "algorithm_with_required_param": AlgorithmSpecification(
            name="algorithm_with_required_param",
            desc="algorithm_with_required_param",
            label="algorithm_with_required_param",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                ),
            ),
            parameters={
                "required_param": ParameterSpecification(
                    label="required_param",
                    desc="required_param",
                    types=[ParameterType.REAL],
                    required=True,
                    multiple=False,
                ),
                "optional_param": ParameterSpecification(
                    label="optional_param",
                    desc="optional_param",
                    types=[ParameterType.REAL],
                    required=False,
                    multiple=False,
                ),
            },
        ),
        "algorithm_with_many_params": AlgorithmSpecification(
            name="algorithm_with_many_params",
            desc="algorithm_with_many_params",
            label="algorithm_with_many_params",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="x",
                    desc="x",
                    types=[InputDataType.INT, InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    required=False,
                    multiple=True,
                ),
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                ),
            ),
            parameters={
                "int_parameter_with_min_max": ParameterSpecification(
                    label="parameter_with_min_max",
                    desc="parameter_with_min_max",
                    types=[ParameterType.INT],
                    required=False,
                    multiple=False,
                    min=2,
                    max=5,
                ),
                "text_parameter": ParameterSpecification(
                    label="text_parameter",
                    desc="text_parameter",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                ),
                "parameter_multiple_true": ParameterSpecification(
                    label="parameter_multiple_true",
                    desc="parameter_multiple_true",
                    types=[ParameterType.INT],
                    required=False,
                    multiple=True,
                ),
                "param_with_enum_type_list": ParameterSpecification(
                    label="param_with_enum_type_list",
                    desc="param_with_enum_type_list",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST,
                        source=["a", "b", "c"],
                    ),
                ),
                "param_with_enum_type_list_multiple_true": ParameterSpecification(
                    label="param_with_enum_type_list_multiple_true",
                    desc="param_with_enum_type_list_multiple_true",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=True,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST,
                        source=["a", "b", "c"],
                    ),
                ),
                "param_with_enum_type_input_var_CDE_enums": ParameterSpecification(
                    label="param_with_enum_type_input_var_CDE_enums",
                    desc="param_with_enum_type_input_var_CDE_enums",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["y"],
                    ),
                ),
                "param_with_enum_type_fixed_var_CDE_enums": ParameterSpecification(
                    label="param_with_enum_type_fixed_var_CDE_enums",
                    desc="param_with_enum_type_fixed_var_CDE_enums",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.FIXED_VAR_CDE_ENUMS,
                        source=["text_cde_categ"],
                    ),
                ),
                "param_with_enum_type_fixed_var_CDE_enums_wrong_CDE": ParameterSpecification(
                    label="param_with_enum_type_fixed_var_CDE_enums_wrong_CDE",
                    desc="param_with_enum_type_fixed_var_CDE_enums_wrong_CDE",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.FIXED_VAR_CDE_ENUMS,
                        source=["non_existing_CDE"],
                    ),
                ),
                "param_with_enum_type_input_var_names": ParameterSpecification(
                    label="param_with_enum_type_input_var_names",
                    desc="param_with_enum_type_input_var_names",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
                ),
                "param_with_type_dict": ParameterSpecification(
                    label="param_with_type_dict",
                    desc="param_with_type_dict",
                    types=[ParameterType.DICT],
                    required=False,
                    multiple=False,
                ),
                "param_with_dict_enums": ParameterSpecification(
                    label="param_with_dict_enums",
                    desc="param_with_dict_enums",
                    types=[ParameterType.DICT],
                    required=False,
                    multiple=False,
                    dict_keys_enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
                    dict_values_enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST,
                        source=["diff", "first", "second"],
                    ),
                ),
            },
        ),
        "algorithm_with_transformer": AlgorithmSpecification(
            name="algorithm_with_transformer",
            desc="algorithm_with_transformer",
            label="algorithm_with_transformer",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.REAL],
                    stattypes=[InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                ),
            ),
        ),
    }


@pytest.fixture(scope="module")
def preprocessing_steps_specs():
    return {
        "transformer_with_real_param": PreprocessingStepSpecification(
            name="transformer_with_real_param",
            desc="transformer_with_real_param",
            label="transformer_with_real_param",
            type=PreprocessingStep.EXAREME3_PREPROCESSING_STEP,
            components=[],
            enabled=True,
            parameters={
                "required_real_param": ParameterSpecification(
                    label="required_real_param",
                    desc="required_real_param",
                    types=[ParameterType.REAL],
                    required=True,
                    multiple=False,
                ),
                "optional_real_param": ParameterSpecification(
                    label="optional_real_param",
                    desc="optional_real_param",
                    types=[ParameterType.REAL],
                    required=False,
                    multiple=False,
                ),
            },
            compatible_algorithms=["algorithm_with_transformer"],
        ),
        "transformer_compatible_with_all_algorithms": PreprocessingStepSpecification(
            name="transformer_compatible_with_all_algorithms",
            desc="transformer_compatible_with_all_algorithms",
            label="transformer_compatible_with_all_algorithms",
            type=PreprocessingStep.EXAREME3_PREPROCESSING_STEP,
            components=[],
            enabled=True,
        ),
    }


def get_parametrization_list_success_cases():
    parametrization_list = [
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1", "sample_dataset2"],
                    y=["int_cde"],
                ),
            ),
            id="multiple datasets",
        ),
        pytest.param(
            "algorithm_with_y_int_and_validation",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1", "sample_dataset2"],
                    validation_datasets=["sample_dataset3"],
                    y=["int_cde"],
                ),
            ),
            id="multiple datasets and validation dataset",
        ),
        pytest.param(
            "algorithm_with_x_int_and_y_text",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    x=["int_cde"],
                    y=["text_cde_categ"],
                ),
            ),
            id="proper variable types",
        ),
        pytest.param(
            "algorithm_with_y_text_multiple_true",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ", "text_cde_non_categ"],
                ),
            ),
            id="variables multiple true",
        ),
        pytest.param(
            "algorithm_with_variable_enumslen",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
            ),
            id="variable enumslen",
        ),
        pytest.param(
            "algorithm_with_y_and_x_optional",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
            ),
            id="variable required false",
        ),
        pytest.param(
            "algorithm_with_y_and_x_optional",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
            ),
            id="variable required false",
        ),
        pytest.param(
            "algorithm_with_required_param",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
                parameters={"required_param": 1},
            ),
            id="parameter required",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": 2},
            ),
            id="parameter min max",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"parameter_multiple_true": [1, 2, 3]},
            ),
            id="parameter multiple true",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_list": "a"},
            ),
            id="parameter enums type list",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_list_multiple_true": ["a", "c"]},
            ),
            id="parameter enums type list and multiple true",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_input_var_CDE_enums": "male"},
            ),
            id="parameter enums type input_var_CDE_enums",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_fixed_var_CDE_enums": "female"},
            ),
            id="parameter enums type fixed_var_CDE_enums",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_input_var_names": "text_cde_categ"},
            ),
            id="parameter enums type input_var_names",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_type_dict": {"sample_key": "sample_value"}},
            ),
            id="Parameter with type dict.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    x=["int_cde", "real_cde"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_dict_enums": {
                        "text_cde_categ": "first",
                        "int_cde": "diff",
                        "real_cde": "diff",
                    }
                },
            ),
            id="Parameter with dict enums.",
        ),
        pytest.param(
            "algorithm_with_transformer",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["real_cde"],
                ),
                preprocessing={
                    "transformer_with_real_param": {"required_real_param": 10.4}
                },
            ),
            id="Algorithm with transformer.",
        ),
        pytest.param(
            "algorithm_with_transformer",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["real_cde"],
                ),
                preprocessing={"transformer_compatible_with_all_algorithms": {}},
            ),
            id="Algorithm with transformer that is compatible with all algorithms.",
        ),
    ]
    return parametrization_list


class WorkerInfo:
    def __init__(self, id):
        self.id = id


mocked_worker_info = WorkerInfo(
    id="globalworker",
)


@pytest.mark.parametrize(
    "algorithm_name, request_dto", get_parametrization_list_success_cases()
)
def test_validate_algorithm_success(
    algorithm_name,
    request_dto,
    worker_landscape_aggregator,
    algorithms_specs,
    preprocessing_steps_specs,
):
    with patch.object(
        worker_landscape_aggregator,
        "get_global_worker",
        return_value=mocked_worker_info,
    ):
        validate_algorithm_request(
            algorithm_name=algorithm_name,
            algorithm_request_dto=request_dto,
            algorithms_specs=algorithms_specs,
            preprocessing_steps_specs=preprocessing_steps_specs,
            worker_landscape_aggregator=worker_landscape_aggregator,
            smpc_enabled=False,
            smpc_optional=False,
        )


def get_parametrization_list_exception_cases():
    parametrization_list = [
        pytest.param(
            "non_existing_algorithm",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                ),
            ),
            (BadRequest, "Algorithm .* does not exist."),
            id="Algorithm does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["non_existing_dataset"],
                    y=["int_cde"],
                ),
            ),
            (
                BadUserInput,
                "Datasets:.* could not be found for data_model:.*",
            ),
            id="Dataset does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int_and_validation",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    validation_datasets=["non_existing_dataset"],
                    y=["int_cde"],
                ),
            ),
            (
                BadUserInput,
                "Validation Datasets:.* could not be found for data_model:.*",
            ),
            id="validation dataset does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset"],
                    y=["int_cde"],
                ),
            ),
            (
                BadUserInput,
                "Datasets:.* could not be found for data_model:.*",
            ),
            id="Dataset does not exist in specified data model.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="non_existing_data_model:0.1",
                    datasets=["sample_dataset"],
                    y=["int_cde"],
                ),
            ),
            (BadUserInput, "Data model .* does not exist."),
            id="Data model does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                ),
            ),
            (BadUserInput, "Inputdata .* should be provided."),
            id="Inputdata not provided.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ", "real_cde"],
                ),
            ),
            (BadUserInput, "Inputdata .* cannot have multiple values."),
            id="Inputdata variable with multiple false given list.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["non_existing"],
                ),
            ),
            (BadUserInput, "The CDE .* does not exist in the data model provided."),
            id="CDE does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
            ),
            (BadUserInput, "The CDE .* doesn't have one of the allowed types .*"),
            id="Inputdata variable wrong CDE type.",
        ),
        pytest.param(
            "algorithm_with_y_text_categ",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_non_categ"],
                ),
            ),
            (BadUserInput, "The CDE .* should be categorical."),
            id="Inputdata variable requires categorical CDE given non categorical.",
        ),
        pytest.param(
            "algorithm_with_y_text_non_categ",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
            ),
            (BadUserInput, "The CDE .* should NOT be categorical."),
            id="Inputdata variable requires non categorical CDE given categorical.",
        ),
        pytest.param(
            "algorithm_with_variable_enumslen",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_3_enums"],
                ),
            ),
            (BadUserInput, "The CDE .* should have .* enumerations."),
            id="Inputdata variable requires 2 enumerations, CDE has 3 enums.",
        ),
        pytest.param(
            "algorithm_with_required_param",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
            ),
            (BadUserInput, "Algorithm parameters not provided."),
            id="Required algorithm parameter not provided.",
        ),
        pytest.param(
            "algorithm_with_required_param",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
                parameters={"optional_param": 1},
            ),
            (BadUserInput, "Parameter .* should not be blank."),
            id="Required parameter not provided.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"parameter_multiple_true": 2},
            ),
            (BadUserInput, "Parameter .* should be a list."),
            id="Parameter with multiple=true given singular value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"text_parameter": 1},
            ),
            (BadUserInput, "Parameter .* values should be of types.*"),
            id="Parameter of type text given int value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": "text_value"},
            ),
            (BadUserInput, "Parameter .* values should be of types.*"),
            id="Parameter of type int given text value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_type_dict": "text_value"},
            ),
            (BadUserInput, "Parameter .* values should be of types.*"),
            id="Parameter of type dict given wrong value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": [1, 2, 3]},
            ),
            (BadUserInput, "Parameter .* values should be of types.*"),
            id="Parameter of type int with multiple=false given list of ints.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_list": "non_existing_enum"},
            ),
            (BadUserInput, "Parameter .* values should be one of the following: .*"),
            id="Parameter with enumerations of type list given non existing enum.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": 1},
            ),
            (BadUserInput, "Parameter .* values should be greater than .*"),
            id="Parameter with min max given a lesser value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": 10},
            ),
            (BadUserInput, "Parameter .* values should be at most equal to .*"),
            id="Parameter with min max given a greater value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_enum_type_input_var_CDE_enums": "non_existing_enum",
                },
            ),
            (
                BadUserInput,
                "Parameter's .* enums, that are taken from the CDE .* given in inputdata .* variable, should be one of the following: .*",
            ),
            id="Parameter with enumerations of type 'input_var_CDE_enums' given non existing enum.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_enum_type_fixed_var_CDE_enums_wrong_CDE": "male",
                },
            ),
            (
                ValueError,
                "Parameter's .* enums source .* does not exist in the data model provided.",
            ),
            id="Parameter with enumerations of type 'fixed_var_CDE_enums' has, in the algorithm specification, a CDE that doesn't exist.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_enum_type_fixed_var_CDE_enums": "non_existing_enum",
                },
            ),
            (
                BadUserInput,
                "Parameter's .* enums, that are taken from the CDE .*, should be one of the following: .*",
            ),
            id="Parameter with enumerations of type 'fixed_var_CDE_enums' given non existing enum.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_enum_type_input_var_names": "text_cde_non_categ",
                },
            ),
            (
                BadUserInput,
                "Parameter's .* enums, that are taken from inputdata .* var names, should be one of the following: .*",
            ),
            id="Parameter with enumerations of type 'input_var_names' given non existing enum.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_dict_enums": {
                        "not_selected_CDE": "first",
                    }
                },
            ),
            (
                BadUserInput,
                "Parameter's .* enums, that are taken from inputdata .* var names, should be one of the following: .*",
            ),
            id="Parameter with 'dict_keys_enums' given wrong key enum.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_dict_enums": {
                        "text_cde_categ": "non_existing_enum",
                    }
                },
            ),
            (
                BadUserInput,
                "Parameter .* values should be one of the following: .*",
            ),
            id="Parameter with 'dict_keys_enums' given wrong value enum.",
        ),
        pytest.param(
            "algorithm_with_transformer",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["real_cde"],
                ),
                preprocessing={"non_existing_transformer": {"real_param": 10.1}},
            ),
            (BadUserInput, "Transformer .* does not exist."),
            id="Transformer does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
                preprocessing={"transformer_with_real_param": {"real_param": 10.1}},
            ),
            (
                BadUserInput,
                "Transformer .* is not available for algorithm .*",
            ),
            id="Transformer provided to incompatible algorithm.",
        ),
        pytest.param(
            "algorithm_with_transformer",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["real_cde"],
                ),
                preprocessing={
                    "transformer_with_real_param": {"optional_real_param": 1}
                },
            ),
            (BadUserInput, "Parameter .* should not be blank."),
            id="Bad parameter input in transformer.",
        ),
        pytest.param(
            "disabled_algorithm",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["real_cde"],
                ),
            ),
            (BadRequest, "Algorithm .* does not exist."),
            id="Disabled algorithm.",
        ),
        pytest.param(
            "algorithm_with_transformer",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["real_cde"],
                ),
                preprocessing={"disabled_transformer": {"real_param": 10.1}},
            ),
            (BadUserInput, "Transformer .* does not exist."),
            id="Disabled Transformer.",
        ),
        pytest.param(
            "algorithm_with_required_param",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
                parameters={
                    "required_param": 1,
                    "non_existing_param": 1,
                },
            ),
            (
                BadUserInput,
                "Parameter .* does not exist in the algorithm specification.",
            ),
            id="non existing parameter provided",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
                flags={"non_existing_flag": True},
            ),
            (
                BadUserInput,
                "Flag .* does not exist in the specifications.",
            ),
            id="non existing flag provided",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
                flags={"smpc": 2},
            ),
            (
                BadUserInput,
                "Flag .* should have a boolean value.",
            ),
            id="flag does not have boolean value",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1", "sample_dataset2"],
                    validation_datasets=["sample_dataset3"],
                    y=["int_cde"],
                ),
            ),
            (
                BadUserInput,
                "The algorithm does not have a validation flow, but 'validation_datasets' were provided in the 'inputdata'.",
            ),
            id="Validation datasets on algorithm without validation",
        ),
        pytest.param(
            "algorithm_with_y_int_and_validation",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1", "sample_dataset2"],
                    y=["int_cde"],
                ),
            ),
            (
                BadUserInput,
                "The algorithm requires 'validation_datasets', in the 'inputdata', but none were provided.",
            ),
            id="Missing validation datasets on algorithm validation",
        ),
    ]
    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dto, exception", get_parametrization_list_exception_cases()
)
def test_validate_algorithm_exceptions(
    algorithm_name,
    request_dto,
    exception,
    algorithms_specs,
    preprocessing_steps_specs,
    worker_landscape_aggregator,
):
    exception_type, exception_message = exception
    with pytest.raises(exception_type, match=exception_message):
        with patch.object(
            worker_landscape_aggregator,
            "get_global_worker",
            return_value=mocked_worker_info,
        ):
            validate_algorithm_request(
                algorithm_name=algorithm_name,
                algorithm_request_dto=request_dto,
                algorithms_specs=algorithms_specs,
                preprocessing_steps_specs=preprocessing_steps_specs,
                worker_landscape_aggregator=worker_landscape_aggregator,
                smpc_enabled=False,
                smpc_optional=False,
            )
