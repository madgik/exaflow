from unittest.mock import patch

import pytest

import exaflow.controller.services.api.algorithm_request_validator as algorithm_request_validator
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
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
from exaflow.algorithms.specifications import PreprocessingStepOrder
from exaflow.algorithms.specifications import PreprocessingStepSpecification
from exaflow.algorithms.specifications import PreprocessingStepType
from exaflow.controller import DeploymentType
from exaflow.controller import logger as ctrl_logger
from exaflow.controller.services.api.algorithm_request_dtos import AlgorithmInputDataDTO
from exaflow.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exaflow.controller.services.api.algorithm_request_dtos import (
    AlgorithmRequestSystemFlags,
)
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
        "algorithm_with_x_single_and_enum_param": AlgorithmSpecification(
            name="algorithm_with_x_single_and_enum_param",
            desc="algorithm_with_x_single_and_enum_param",
            label="algorithm_with_x_single_and_enum_param",
            enabled=True,
            type=AlgorithmType.EXAREME3,
            components=[],
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="x",
                    desc="x",
                    types=[InputDataType.INT, InputDataType.TEXT],
                    stattypes=[InputDataStatType.NOMINAL, InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                ),
                y=InputDataSpecification(
                    label="y",
                    desc="y",
                    types=[InputDataType.TEXT, InputDataType.REAL, InputDataType.INT],
                    stattypes=[InputDataStatType.NOMINAL, InputDataStatType.NUMERICAL],
                    required=False,
                    multiple=True,
                ),
            ),
            parameters={
                "param_with_enum_type_input_var_CDE_enums_x": ParameterSpecification(
                    label="param_with_enum_type_input_var_CDE_enums_x",
                    desc="param_with_enum_type_input_var_CDE_enums_x",
                    types=[ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["x"],
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
            type=PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
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
        ),
        "transformer_compatible_with_all_algorithms": PreprocessingStepSpecification(
            name="transformer_compatible_with_all_algorithms",
            desc="transformer_compatible_with_all_algorithms",
            label="transformer_compatible_with_all_algorithms",
            type=PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
            components=[],
            enabled=True,
        ),
        "rename_to_real": PreprocessingStepSpecification(
            name="rename_to_real",
            desc="rename_to_real",
            label="rename_to_real",
            type=PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
            components=[],
            enabled=True,
        ),
        "step_using_transformed_inputdata": PreprocessingStepSpecification(
            name="step_using_transformed_inputdata",
            desc="step_using_transformed_inputdata",
            label="step_using_transformed_inputdata",
            type=PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
            components=[],
            enabled=True,
            parameters={
                "selected_var": ParameterSpecification(
                    label="selected_var",
                    desc="selected_var",
                    types=[ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUT_VAR_NAMES,
                        source=["y"],
                    ),
                )
            },
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
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                preprocessing={"rename_to_real": {}},
            ),
            id="Algorithm inputdata x/y validation uses preprocessed variables.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                preprocessing={
                    "rename_to_real": {},
                    "step_using_transformed_inputdata": {"selected_var": "real_cde"},
                },
            ),
            id="Preprocessing params are validated using transformed inputdata from previous steps.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                preprocessing={
                    "step_using_transformed_inputdata": {"selected_var": "real_cde"},
                    "rename_to_real": {},
                },
            ),
            id="Preprocessing execution order is derived from step order enum, not request map order.",
        ),
    ]
    return parametrization_list


class WorkerInfo:
    def __init__(self, id):
        self.id = id


mocked_worker_info = WorkerInfo(
    id="globalworker",
)


class RenameYToRealStep(PreprocessingStep):
    def __init__(self, *, params):
        self._params = params

    @classmethod
    def get_specification(cls):
        return PreprocessingStepSpecification(
            name="rename_to_real",
            desc="rename_to_real",
            label="rename_to_real",
            enabled=True,
            order=PreprocessingStepOrder.FIRST,
        )

    def validate_params(self, *, inputdata, metadata):
        return None

    def transform_inputdata_variables(self, *, x, y):
        return x, ["real_cde"]

    def transform_metadata(self, *, metadata):
        return metadata

    def transform_data(self, *, data):
        return data

    @classmethod
    def required_input_variables(cls):
        return []


class NoopPreprocessingStep(PreprocessingStep):
    def __init__(self, *, params):
        self._params = params

    @classmethod
    def get_specification(cls):
        return PreprocessingStepSpecification(
            name="noop_preprocessing_step",
            desc="noop_preprocessing_step",
            label="noop_preprocessing_step",
            enabled=True,
            order=PreprocessingStepOrder.FOURTH,
        )

    def validate_params(self, *, inputdata, metadata):
        return None

    def transform_inputdata_variables(self, *, x, y):
        return x, y

    def transform_metadata(self, *, metadata):
        return metadata

    def transform_data(self, *, data):
        return data

    @classmethod
    def required_input_variables(cls):
        return []


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
    with patch.dict(
        algorithm_request_validator.exareme3_preprocessing_step_classes,
        {
            "rename_to_real": RenameYToRealStep,
            "transformer_with_real_param": NoopPreprocessingStep,
            "transformer_compatible_with_all_algorithms": NoopPreprocessingStep,
            "step_using_transformed_inputdata": NoopPreprocessingStep,
        },
        clear=False,
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


def test_validate_algorithm_preprocessing_raises_when_implementation_is_missing(
    worker_landscape_aggregator,
    algorithms_specs,
    preprocessing_steps_specs,
):
    request_dto = AlgorithmRequestDTO(
        inputdata=AlgorithmInputDataDTO(
            data_model="data_model_with_all_cde_types:0.1",
            datasets=["sample_dataset1"],
            y=["real_cde"],
        ),
        preprocessing={"transformer_with_real_param": {"required_real_param": 10.4}},
    )

    with patch.dict(
        algorithm_request_validator.exareme3_preprocessing_step_classes,
        {"transformer_with_real_param": None},
        clear=False,
    ):
        with patch.object(
            worker_landscape_aggregator,
            "get_global_worker",
            return_value=mocked_worker_info,
        ):
            with pytest.raises(
                BadUserInput,
                match="Preprocessing step .* implementation was not found.",
            ):
                validate_algorithm_request(
                    algorithm_name="algorithm_with_transformer",
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
            "algorithm_with_x_single_and_enum_param",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    x=["int_cde_categ"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_enum_type_input_var_CDE_enums_x": "non_existing_enum",
                },
            ),
            (
                BadUserInput,
                "Parameter's .* enums, that are taken from the CDE .* given in inputdata .* variable, should be one of the following: .*",
            ),
            id="Parameter with enumerations of type 'input_var_CDE_enums' from x given non existing enum.",
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
                "Parameter .* does not exist in the algorithm specification.",
            ),
            id="Transformer parameter does not exist.",
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


def test_validate_algorithm_inputdata_returns_when_values_and_spec_are_missing(
    worker_landscape_aggregator,
):
    data_model_cdes = worker_landscape_aggregator.get_cdes(
        "data_model_with_all_cde_types:0.1"
    )
    algorithm_request_validator._validate_algorithm_inputdata(
        inputdata_values=None,
        inputdata_spec=None,
        data_model_cdes=data_model_cdes,
    )


def test_validate_inputdata_values_quantity_raises_for_non_list():
    with pytest.raises(BadRequest, match="Inputdata .* should be a list."):
        algorithm_request_validator._validate_inputdata_values_quantity(
            inputdata_value="not_a_list",
            inputdata_spec=InputDataSpecification(
                label="y",
                desc="y",
                types=[InputDataType.REAL],
                stattypes=[InputDataStatType.NUMERICAL],
                required=True,
                multiple=False,
            ),
        )


def test_validate_param_enums_input_var_cde_enums_with_x_source_success(
    worker_landscape_aggregator,
):
    data_model_cdes = worker_landscape_aggregator.get_cdes(
        "data_model_with_all_cde_types:0.1"
    )
    inputdata = AlgorithmInputDataDTO(
        data_model="data_model_with_all_cde_types:0.1",
        datasets=["sample_dataset1"],
        x=["int_cde_categ"],
        y=["text_cde_categ"],
    )
    algorithm_request_validator._validate_param_enums(
        parameter_value="1",
        parameter_spec_enums=ParameterEnumSpecification(
            type=ParameterEnumType.INPUT_VAR_CDE_ENUMS,
            source=["x"],
        ),
        parameter_spec_label="param",
        inputdata=inputdata,
        data_model_cdes=data_model_cdes,
    )


def test_validate_flags_with_smpc_calls_validate_smpc_usage():
    with patch.object(
        algorithm_request_validator,
        "validate_smpc_usage",
    ) as validate_smpc_usage_mock:
        algorithm_request_validator._validate_flags(
            flags={AlgorithmRequestSystemFlags.SMPC: True},
            smpc_enabled=True,
            smpc_optional=True,
        )

    validate_smpc_usage_mock.assert_called_once_with(True, True, True)
