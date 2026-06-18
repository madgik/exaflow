import numbers
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from exaflow import exareme3_preprocessing_step_classes
from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterDictValueType
from exaflow.algorithms.specifications import ParameterEnumSpecification
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import PreprocessingOutputType
from exaflow.algorithms.specifications import PreprocessingStepSpecification
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.controller.services.api.algorithm_spec_dtos import ParameterEnumType
from exaflow.controller.services.api.algorithm_spec_dtos import ParameterType
from exaflow.controller.services.api.analysis_request_dtos import AnalysisInputDataDTO
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.api.analysis_request_dtos import (
    AnalysisRequestSystemFlags,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.data_filters import validate_filter
from exaflow.smpc_cluster_communication import validate_smpc_usage
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement


class BadRequest(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def validate_analysis_request(
    analysis_request_dto: AnalysisRequestDTO,
    algorithms_specs: Dict[str, AlgorithmSpecification],
    preprocessing_steps_specs: Dict[str, PreprocessingStepSpecification],
    worker_landscape_aggregator: WorkerLandscapeAggregator,
    smpc_enabled: bool,
    smpc_optional: bool,
):
    algorithm_name = analysis_request_dto.algorithm.name
    algorithm_specs = _get_algorithm_specs(algorithm_name, algorithms_specs)

    (
        training_datasets,
        validation_datasets,
    ) = worker_landscape_aggregator.get_training_and_validation_datasets(
        analysis_request_dto.inputdata.data_model
    )
    data_model_cdes = worker_landscape_aggregator.get_cdes(
        analysis_request_dto.inputdata.data_model
    )
    _validate_analysis_request_body(
        analysis_request_dto=analysis_request_dto,
        algorithm_specs=algorithm_specs,
        preprocessing_steps_specs=preprocessing_steps_specs,
        training_datasets=training_datasets,
        validation_datasets=validation_datasets,
        data_model_cdes=data_model_cdes,
        smpc_enabled=smpc_enabled,
        smpc_optional=smpc_optional,
    )


def _get_algorithm_specs(
    algorithm_name: str,
    algorithms_specs: Dict[str, AlgorithmSpecification],
):
    if algorithm_name not in algorithms_specs.keys():
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")
    return algorithms_specs[algorithm_name]


def _validate_analysis_request_body(
    analysis_request_dto: AnalysisRequestDTO,
    algorithm_specs: AlgorithmSpecification,
    preprocessing_steps_specs: Dict[str, PreprocessingStepSpecification],
    training_datasets: List[str],
    validation_datasets: List[str],
    data_model_cdes: Dict[str, CommonDataElement],
    smpc_enabled: bool,
    smpc_optional: bool,
):
    _validate_inputdata_base(
        inputdata=analysis_request_dto.inputdata,
        training_datasets=training_datasets,
        algorithm_specification_requires_validation_datasets=algorithm_specs.requires_validation_datasets,
        validation_datasets=validation_datasets,
        data_model_cdes=data_model_cdes,
    )

    _validate_required_preprocessing(
        algorithm_specs=algorithm_specs,
        preprocessing=analysis_request_dto.preprocessing,
    )

    transformed_inputdata, transformed_data_model_cdes = (
        _validate_and_apply_preprocessing(
            analysis_request_dto=analysis_request_dto,
            preprocessing_steps_specs=preprocessing_steps_specs,
            data_model_cdes=data_model_cdes,
        )
    )

    _validate_algorithm_inputdatas(
        x=analysis_request_dto.algorithm.x,
        y=analysis_request_dto.algorithm.y,
        algorithm_specs=algorithm_specs,
        data_model_cdes=transformed_data_model_cdes,
    )

    _validate_parameters(
        analysis_request_dto.algorithm.parameters,
        algorithm_specs.parameters,
        transformed_inputdata,
        data_model_cdes=transformed_data_model_cdes,
        algorithm_x=analysis_request_dto.algorithm.x,
        algorithm_y=analysis_request_dto.algorithm.y,
    )

    _validate_flags(
        flags=analysis_request_dto.flags,
        smpc_enabled=smpc_enabled,
        smpc_optional=smpc_optional,
    )


def _validate_required_preprocessing(
    algorithm_specs: AlgorithmSpecification,
    preprocessing: Optional[List[Any]],
):
    requested_preprocessing = {step.name for step in (preprocessing or [])}
    missing_required_preprocessing = [
        name
        for name in algorithm_specs.required_preprocessing
        if name not in requested_preprocessing
    ]
    if missing_required_preprocessing:
        raise BadUserInput(
            f"Algorithm '{algorithm_specs.name}' requires preprocessing steps: "
            f"{missing_required_preprocessing}."
        )


def _validate_inputdata_base(
    inputdata: AnalysisInputDataDTO,
    training_datasets: List[str],
    algorithm_specification_requires_validation_datasets: bool,
    validation_datasets: List[str],
    data_model_cdes: Dict[str, CommonDataElement],
):
    _validate_inputdata_training_datasets(
        requested_data_model=inputdata.data_model,
        requested_training_datasets=inputdata.datasets,
        training_datasets=training_datasets,
    )
    _validate_inputdata_validation_datasets(
        requested_data_model=inputdata.data_model,
        requested_validation_datasets=inputdata.validation_datasets,
        algorithm_specification_requires_validation_datasets=algorithm_specification_requires_validation_datasets,
        validation_datasets=validation_datasets,
    )
    _validate_inputdata_filter(inputdata.data_model, inputdata.filters, data_model_cdes)
    _validate_source_variables(inputdata.variables, data_model_cdes)


def _validate_and_apply_preprocessing(
    analysis_request_dto: AnalysisRequestDTO,
    preprocessing_steps_specs: Dict[str, PreprocessingStepSpecification],
    data_model_cdes: Dict[str, CommonDataElement],
):
    transformed_inputdata = _build_source_inputdata(analysis_request_dto)
    transformed_data_model_cdes = _select_available_cdes(
        analysis_request_dto.inputdata.variables, data_model_cdes
    )
    transformed_metadata = _convert_data_model_cdes_to_metadata(
        transformed_data_model_cdes
    )

    if not analysis_request_dto.preprocessing:
        return transformed_inputdata, transformed_data_model_cdes

    for step in analysis_request_dto.preprocessing:
        name = step.name
        params = step.parameters
        if name not in preprocessing_steps_specs.keys():
            raise BadUserInput(f"Transformer '{name}' does not exist.")

        preprocessing_step_spec = preprocessing_steps_specs[name]
        preprocessing_step_cls = exareme3_preprocessing_step_classes.get(name)
        step_data_model_cdes = transformed_data_model_cdes
        step_metadata = transformed_metadata
        if preprocessing_step_cls:
            required_data_model_cdes = {}
            required_metadata = {}
            for variable in preprocessing_step_cls.required_input_variables():
                if variable in data_model_cdes:
                    required_data_model_cdes[variable] = data_model_cdes[variable]
                    required_metadata[variable] = data_model_cdes[variable].model_dump()
            step_data_model_cdes = {
                **transformed_data_model_cdes,
                **required_data_model_cdes,
            }
            step_metadata = {
                **transformed_metadata,
                **required_metadata,
            }

        _validate_parameters(
            parameters=params,
            parameters_specs=preprocessing_step_spec.parameters,
            inputdata=transformed_inputdata,
            data_model_cdes=step_data_model_cdes,
        )
        _validate_preprocessing_output_name(
            preprocessing_step_spec=preprocessing_step_spec,
            params=params,
            data_model_cdes={**data_model_cdes, **transformed_data_model_cdes},
        )

        if not preprocessing_step_cls:
            raise BadUserInput(
                f"Preprocessing step '{name}' is enabled but its implementation was not found."
            )

        preprocessing_step = preprocessing_step_cls(params=params)
        preprocessing_step.validate_params(
            inputdata=transformed_inputdata,
            metadata=step_metadata,
        )
        transformed_variables = preprocessing_step.transform_variables(
            variables=list(transformed_inputdata.variables),
        )
        transformed_inputdata = transformed_inputdata.model_copy(
            update={"variables": transformed_variables}
        )
        transformed_metadata = preprocessing_step.transform_metadata(
            metadata=transformed_metadata,
        )
        _derive_preprocessing_output_metadata(
            preprocessing_step_spec=preprocessing_step_spec,
            params=params,
            metadata=transformed_metadata,
        )
        transformed_data_model_cdes = _convert_metadata_to_data_model_cdes(
            transformed_metadata
        )

    return transformed_inputdata, transformed_data_model_cdes


def _build_source_inputdata(
    analysis_request_dto: AnalysisRequestDTO,
) -> Inputdata:
    return Inputdata(
        data_model=analysis_request_dto.inputdata.data_model,
        datasets=analysis_request_dto.inputdata.datasets,
        validation_datasets=analysis_request_dto.inputdata.validation_datasets,
        filters=analysis_request_dto.inputdata.filters,
        variables=analysis_request_dto.inputdata.variables,
    )


def _validate_source_variables(
    variables: List[str],
    data_model_cdes: Dict[str, CommonDataElement],
) -> None:
    if not variables:
        raise BadUserInput("Inputdata 'variables' should be provided.")
    duplicate_variables = sorted(
        {variable for variable in variables if variables.count(variable) > 1}
    )
    if duplicate_variables:
        raise BadUserInput(
            f"Inputdata 'variables' should not contain duplicate variables: {duplicate_variables}."
        )
    missing_variables = [
        variable for variable in variables if variable not in data_model_cdes
    ]
    if missing_variables:
        raise BadUserInput(
            f"Inputdata 'variables' contain CDEs that do not exist in the data model provided: {missing_variables}."
        )


def _select_available_cdes(
    variables: List[str],
    data_model_cdes: Dict[str, CommonDataElement],
) -> Dict[str, CommonDataElement]:
    return {variable: data_model_cdes[variable] for variable in variables}


def _validate_preprocessing_output_name(
    *,
    preprocessing_step_spec: PreprocessingStepSpecification,
    params: Dict[str, Any],
    data_model_cdes: Dict[str, CommonDataElement],
) -> None:
    output = preprocessing_step_spec.output
    if (
        not output
        or output.type != PreprocessingOutputType.NEW_CATEGORICAL_COLUMN
        or not output.code_parameter
    ):
        return

    code = params.get(output.code_parameter)
    if code in data_model_cdes:
        raise BadUserInput(
            f"Preprocessing step '{preprocessing_step_spec.name}' cannot create CDE '{code}' because it already exists."
        )


def _derive_preprocessing_output_metadata(
    *,
    preprocessing_step_spec: PreprocessingStepSpecification,
    params: Dict[str, Any],
    metadata: Dict[str, dict],
) -> None:
    output = preprocessing_step_spec.output
    if (
        not output
        or output.type != PreprocessingOutputType.NEW_CATEGORICAL_COLUMN
        or not output.code_parameter
    ):
        return

    code = params[output.code_parameter]
    if code in metadata:
        return

    rules = params["rules"]
    default_enumeration = params.get("default_enumeration")
    enumerations = {key: key for key in rules.keys()}
    if default_enumeration:
        enumerations[default_enumeration] = default_enumeration
    metadata[code] = {
        "code": code,
        "label": code,
        "sql_type": "text",
        "is_categorical": True,
        "enumerations": enumerations,
    }


def _convert_data_model_cdes_to_metadata(
    data_model_cdes: Dict[str, CommonDataElement],
) -> Dict[str, dict]:
    return {
        cde_name: cde_metadata.model_dump()
        for cde_name, cde_metadata in data_model_cdes.items()
    }


def _convert_metadata_to_data_model_cdes(
    metadata: Dict[str, dict],
) -> Dict[str, CommonDataElement]:
    transformed = {}
    for cde_name, cde_metadata in metadata.items():
        serialized_cde_metadata = dict(cde_metadata)
        serialized_cde_metadata["code"] = cde_name
        serialized_cde_metadata.setdefault("label", cde_name)
        transformed[cde_name] = CommonDataElement.model_validate(
            serialized_cde_metadata
        )
    return transformed


def _validate_inputdata_training_datasets(
    requested_data_model: str,
    requested_training_datasets: List[str],
    training_datasets: List[str],
):
    """
    Validates that the dataset values exist.
    """
    non_existing_datasets = [
        dataset
        for dataset in requested_training_datasets
        if dataset not in training_datasets
    ]
    if non_existing_datasets:
        raise BadUserInput(
            f"Datasets:'{non_existing_datasets}' could not be found for data_model:{requested_data_model}"
        )


def _validate_inputdata_validation_datasets(
    requested_data_model: str,
    requested_validation_datasets: List[str],
    algorithm_specification_requires_validation_datasets: bool,
    validation_datasets: List[str],
):
    """
    Validates that the validation dataset values exist.
    """
    if (
        not algorithm_specification_requires_validation_datasets
        and requested_validation_datasets
    ):
        raise BadUserInput(
            "The algorithm does not have a validation flow, but 'validation_datasets' were provided in the 'inputdata'."
        )
    elif (
        algorithm_specification_requires_validation_datasets
        and not requested_validation_datasets
    ):
        raise BadUserInput(
            "The algorithm requires 'validation_datasets', in the 'inputdata', but none were provided."
        )

    if not requested_validation_datasets:
        return

    non_existing_datasets = [
        dataset
        for dataset in requested_validation_datasets
        if dataset not in validation_datasets
    ]
    if non_existing_datasets:
        raise BadUserInput(
            f"Validation Datasets:'{non_existing_datasets}' could not be found for data_model:{requested_data_model}"
        )


def _validate_inputdata_filter(data_model, filter, data_model_cdes):
    """
    Validates that the filter provided have the correct format
    following: https://querybuilder.js.org/
    """
    validate_filter(data_model, filter, data_model_cdes)


def _validate_algorithm_inputdatas(
    x: Optional[List[str]],
    y: Optional[List[str]],
    algorithm_specs: AlgorithmSpecification,
    data_model_cdes: Dict[str, CommonDataElement],
):
    _validate_algorithm_variable_uniqueness(x=x, y=y)

    if algorithm_specs.x:
        _validate_algorithm_inputdata(x, algorithm_specs.x, data_model_cdes)
    if algorithm_specs.y:
        _validate_algorithm_inputdata(y, algorithm_specs.y, data_model_cdes)


def _validate_algorithm_variable_uniqueness(
    x: Optional[List[str]],
    y: Optional[List[str]],
):
    x_values = x or []
    y_values = y or []

    if len(x_values) != len(set(x_values)):
        raise BadUserInput("Algorithm 'x' should not contain duplicate variables.")

    if len(y_values) != len(set(y_values)):
        raise BadUserInput("Algorithm 'y' should not contain duplicate variables.")

    overlap = set(x_values).intersection(y_values)
    if overlap:
        raise BadUserInput(
            "Algorithm 'x' and 'y' should not contain the same variables."
        )


def _validate_algorithm_inputdata(
    inputdata_values: Optional[List[str]],
    inputdata_spec: InputDataSpecification,
    data_model_cdes: Dict[str, CommonDataElement],
):
    if not inputdata_values and not inputdata_spec:
        return

    if not inputdata_values:
        effective_min = (
            inputdata_spec.min_count
            if inputdata_spec.min_count is not None
            else (1 if inputdata_spec.required else 0)
        )
        if effective_min > 0:
            raise BadUserInput(
                f"Algorithm input '{inputdata_spec.label}' should be provided."
            )
        else:
            return

    _validate_inputdata_values_quantity(inputdata_values, inputdata_spec)

    for inputdata_value in inputdata_values:
        _validate_inputdata_value(inputdata_value, inputdata_spec, data_model_cdes)


def _validate_inputdata_values_quantity(
    inputdata_value: Any, inputdata_spec: InputDataSpecification
):
    if not isinstance(inputdata_value, list):
        raise BadRequest(f"Algorithm input '{inputdata_spec.label}' should be a list.")

    size = len(inputdata_value)
    effective_min = (
        inputdata_spec.min_count
        if inputdata_spec.min_count is not None
        else (1 if inputdata_spec.required else 0)
    )
    if size < effective_min:
        raise BadUserInput(
            f"Algorithm input '{inputdata_spec.label}' should include at least {effective_min} values."
        )

    if inputdata_spec.max_count is not None and size > inputdata_spec.max_count:
        raise BadUserInput(
            f"Algorithm input '{inputdata_spec.label}' should include at most {inputdata_spec.max_count} values."
        )


def _validate_inputdata_value(
    inputdata_value: str,
    inputdata_specs: InputDataSpecification,
    data_model_cdes: Dict[str, CommonDataElement],
):
    inputdata_value_metadata = _get_cde_metadata(inputdata_value, data_model_cdes)
    _validate_inputdata_types(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )
    _validate_inputdata_stattypes(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )


def _get_cde_metadata(cde, data_model_cdes):
    if cde not in data_model_cdes.keys():
        raise BadUserInput(
            f"The CDE '{cde}' does not exist in the data model provided."
        )
    return data_model_cdes[cde]


def _validate_inputdata_types(
    inputdata_value: str,
    inputdata_specs: InputDataSpecification,
    inputdata_value_metadata: CommonDataElement,
):
    dtype = InputDataType(inputdata_value_metadata.sql_type)
    dtypes = inputdata_specs.types
    if dtype in dtypes:
        return
    if InputDataType.REAL in dtypes and dtype in (
        InputDataType.INT,
        InputDataType.REAL,
    ):
        return
    raise BadUserInput(
        f"The CDE '{inputdata_value}', of algorithm input '{inputdata_specs.label}', "
        f"doesn't have one of the allowed types "
        f"'{inputdata_specs.types}'."
    )


def _validate_inputdata_stattypes(
    inputdata_value: str,
    inputdata_specs: InputDataSpecification,
    inputdata_value_metadata: CommonDataElement,
):
    can_be_numerical = InputDataStatType.NUMERICAL in inputdata_specs.stattypes
    can_be_nominal = InputDataStatType.NOMINAL in inputdata_specs.stattypes
    if not inputdata_value_metadata.is_categorical and not can_be_numerical:
        raise BadUserInput(
            f"The CDE '{inputdata_value}', of algorithm input '{inputdata_specs.label}', "
            f"should be categorical."
        )
    if inputdata_value_metadata.is_categorical and not can_be_nominal:
        raise BadUserInput(
            f"The CDE '{inputdata_value}', of algorithm input '{inputdata_specs.label}', "
            f"should NOT be categorical."
        )


def _validate_parameters(
    parameters: Optional[Dict[str, Any]],
    parameters_specs: Optional[Dict[str, ParameterSpecification]],
    inputdata: Inputdata,
    data_model_cdes: Dict[str, CommonDataElement],
    algorithm_x: Optional[List[str]] = None,
    algorithm_y: Optional[List[str]] = None,
):
    """
    If the algorithm has parameters,
    it validates that they follow the algorithm specs.
    """
    _validate_parameters_are_in_the_specs(parameters, parameters_specs)

    if parameters_specs is None:
        return

    for parameter_name, parameter_spec in parameters_specs.items():
        if parameter_spec.required:
            if not parameters:
                raise BadUserInput(f"Algorithm parameters not provided.")
            if parameter_name not in parameters.keys():
                raise BadUserInput(f"Parameter '{parameter_name}' should not be blank.")

        if parameters is None or parameter_name not in parameters:
            continue

        parameter_values = parameters.get(parameter_name)
        if parameter_values is None:
            if parameter_spec.required:
                raise BadUserInput(f"Parameter '{parameter_name}' should not be blank.")
            continue

        if isinstance(parameter_values, str) and not parameter_values.strip():
            raise BadUserInput(f"Parameter '{parameter_name}' should not be blank.")

        if isinstance(parameter_values, list) and len(parameter_values) == 0:
            raise BadUserInput(f"Parameter '{parameter_name}' should not be blank.")

        if isinstance(parameter_values, dict) and len(parameter_values) == 0:
            raise BadUserInput(f"Parameter '{parameter_name}' should not be blank.")

        if isinstance(parameter_values, bool):
            _validate_parameter_values(
                parameter_values=parameter_values,
                parameter_spec=parameter_spec,
                inputdata=inputdata,
                data_model_cdes=data_model_cdes,
                algorithm_x=algorithm_x,
                algorithm_y=algorithm_y,
            )
            continue

        if parameter_values:
            _validate_parameter_values(
                parameter_values=parameter_values,
                parameter_spec=parameter_spec,
                inputdata=inputdata,
                data_model_cdes=data_model_cdes,
                algorithm_x=algorithm_x,
                algorithm_y=algorithm_y,
            )


def _validate_parameters_are_in_the_specs(
    parameters: Optional[Dict[str, Any]],
    parameters_specs: Optional[Dict[str, ParameterSpecification]],
):
    if parameters:
        for param_name in parameters.keys():
            if not parameters_specs or param_name not in parameters_specs.keys():
                raise BadUserInput(
                    f"Parameter {param_name} does not exist in the algorithm specification."
                )


def _validate_parameter_values(
    parameter_values: Any,
    parameter_spec: ParameterSpecification,
    inputdata: Inputdata,
    data_model_cdes: Dict[str, CommonDataElement],
    algorithm_x: Optional[List[str]] = None,
    algorithm_y: Optional[List[str]] = None,
):
    if parameter_spec.multiple and not isinstance(parameter_values, list):
        raise BadUserInput(f"Parameter '{parameter_spec.label}' should be a list.")

    if not parameter_spec.multiple:
        parameter_values = [parameter_values]
    for parameter_value in parameter_values:
        _validate_parameter_type(parameter_value, parameter_spec)

        _validate_param_enums(
            parameter_value,
            parameter_spec.enums,
            parameter_spec.label,
            inputdata,
            data_model_cdes,
            algorithm_x,
            algorithm_y,
        )

        _validate_param_dict_enums(
            parameter_value,
            parameter_spec,
            inputdata,
            data_model_cdes,
            algorithm_x,
            algorithm_y,
        )

        _validate_parameter_inside_min_max(parameter_value, parameter_spec)


def _validate_parameter_type(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
):
    exaflowtypes_to_python_types = {
        "text": str,
        "int": int,
        "real": numbers.Real,
        "boolean": bool,
        "dict": dict,
    }

    for param_type in parameter_spec.types:
        if isinstance(
            parameter_value, exaflowtypes_to_python_types.get(param_type.value)
        ):
            return
    else:
        raise BadUserInput(
            f"Parameter '{parameter_spec.label}' values should be of types: {[type.value for type in parameter_spec.types]}."
        )


def _validate_parameter_value_type(
    parameter_value: Any,
    parameter_type: ParameterDictValueType,
    parameter_spec_label: str,
):
    exaflowtypes_to_python_types = {
        "text": str,
        "int": int,
        "real": numbers.Real,
        "boolean": bool,
    }

    if parameter_type == ParameterDictValueType.FILTER:
        return

    if isinstance(parameter_value, exaflowtypes_to_python_types[parameter_type.value]):
        return

    raise BadUserInput(
        f"Parameter '{parameter_spec_label}' dictionary values should be of type: {parameter_type.value}."
    )


def _validate_param_enums_of_type_input_var_names(
    parameter_value: Any,
    parameter_spec_enums: ParameterEnumSpecification,
    parameter_spec_label: str,
    inputdata: Inputdata,
    algorithm_x: Optional[List[str]],
    algorithm_y: Optional[List[str]],
):
    input_var_names_enums = []
    for source in parameter_spec_enums.source:
        if source == "variables":
            input_var_names_enums.extend(inputdata.variables)
        elif source == "x":
            input_var_names_enums.extend(algorithm_x) if algorithm_x else None
        elif source == "y":
            input_var_names_enums.extend(algorithm_y) if algorithm_y else None
        else:
            raise NotImplementedError(
                "Input var names enums source should be 'variables', 'x', or 'y'."
            )

    if parameter_value not in input_var_names_enums:
        raise BadUserInput(
            f"Parameter's '{parameter_spec_label}' enums, that are taken from inputdata {parameter_spec_enums.source} var names, "
            f"should be one of the following: '{input_var_names_enums}'.",
        )


def _validate_param_enums_of_type_fixed_var_CDE_enums(
    parameter_value: Any,
    parameter_spec_enums: ParameterEnumSpecification,
    parameter_spec_label: str,
    data_model_cdes: Dict[str, CommonDataElement],
):
    param_spec_enums_source = parameter_spec_enums.source[
        0
    ]  # Fixed var CDE enums allows only one source value
    if param_spec_enums_source not in data_model_cdes.keys():
        raise ValueError(
            f"Parameter's '{parameter_spec_label}' enums source '{param_spec_enums_source}' does "
            f"not exist in the data model provided."
        )
    fixed_var_CDE_enums = list(
        data_model_cdes[param_spec_enums_source].enumerations.keys()
    )
    if parameter_value not in fixed_var_CDE_enums:
        raise BadUserInput(
            f"Parameter's '{parameter_spec_label}' enums, that are taken from the CDE '{param_spec_enums_source}', "
            f"should be one of the following: '{list(fixed_var_CDE_enums)}'."
        )


def _validate_param_enums_of_type_input_var_CDE_enums(
    parameter_value: Any,
    parameter_spec_enums: ParameterEnumSpecification,
    parameter_spec_label: str,
    data_model_cdes: Dict[str, CommonDataElement],
    algorithm_x: Optional[List[str]],
    algorithm_y: Optional[List[str]],
):
    param_spec_enums_source = parameter_spec_enums.source[
        0
    ]  # Input var CDE enums allows only one source value
    if param_spec_enums_source == "x":
        input_vars = algorithm_x
    elif param_spec_enums_source == "y":
        input_vars = algorithm_y
    else:
        raise NotImplementedError(f"Source should be either 'x' or 'y'.")
    if not input_vars:
        raise BadUserInput(
            f"Parameter's '{parameter_spec_label}' enums source '{param_spec_enums_source}' was not provided."
        )
    input_var = input_vars[0]  # multiple=true is not allowed
    input_var_CDE_enums = data_model_cdes[input_var].enumerations.keys()
    if parameter_value not in input_var_CDE_enums:
        raise BadUserInput(
            f"Parameter's '{parameter_spec_label}' enums, that are taken from the CDE '{input_var}' "
            f"given in inputdata '{parameter_spec_enums.source}' variable, "
            f"should be one of the following: '{list(input_var_CDE_enums)}'."
        )


def _validate_param_enums_of_type_list(
    parameter_value: Any,
    parameter_spec_enums: ParameterEnumSpecification,
    parameter_spec_label: str,
):
    if parameter_value not in parameter_spec_enums.source:
        raise BadUserInput(
            f"Parameter '{parameter_spec_label}' values "
            f"should be one of the following: {parameter_spec_enums.source}. Value provided: '{parameter_value}'."
        )


def _validate_param_enums(
    parameter_value: Any,
    parameter_spec_enums: ParameterEnumSpecification,
    parameter_spec_label: str,
    inputdata: Inputdata,
    data_model_cdes: Dict[str, CommonDataElement],
    algorithm_x: Optional[List[str]] = None,
    algorithm_y: Optional[List[str]] = None,
):
    if parameter_spec_enums is None:
        return

    if parameter_spec_enums.type == ParameterEnumType.LIST:
        _validate_param_enums_of_type_list(
            parameter_value, parameter_spec_enums, parameter_spec_label
        )
    elif parameter_spec_enums.type == ParameterEnumType.INPUT_VAR_CDE_ENUMS:
        _validate_param_enums_of_type_input_var_CDE_enums(
            parameter_value,
            parameter_spec_enums,
            parameter_spec_label,
            data_model_cdes,
            algorithm_x,
            algorithm_y,
        )
    elif parameter_spec_enums.type == ParameterEnumType.FIXED_VAR_CDE_ENUMS:
        _validate_param_enums_of_type_fixed_var_CDE_enums(
            parameter_value, parameter_spec_enums, parameter_spec_label, data_model_cdes
        )
    elif parameter_spec_enums.type == ParameterEnumType.INPUT_VAR_NAMES:
        _validate_param_enums_of_type_input_var_names(
            parameter_value,
            parameter_spec_enums,
            parameter_spec_label,
            inputdata,
            algorithm_x,
            algorithm_y,
        )
    else:
        raise NotImplementedError(
            f"Parameter enum type not supported: '{parameter_spec_enums.type}'."
        )


def _validate_param_dict_enums(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
    inputdata: Inputdata,
    data_model_cdes: Dict[str, CommonDataElement],
    algorithm_x: Optional[List[str]] = None,
    algorithm_y: Optional[List[str]] = None,
):
    if ParameterType.DICT in parameter_spec.types:
        for key in parameter_value.keys():
            _validate_param_enums(
                key,
                parameter_spec.dict_keys_enums,
                parameter_spec.label,
                inputdata,
                data_model_cdes,
                algorithm_x,
                algorithm_y,
            )

        for value in parameter_value.values():
            if parameter_spec.dict_values_type:
                if parameter_spec.dict_values_type == ParameterDictValueType.FILTER:
                    validate_filter(inputdata.data_model, value, data_model_cdes)
                else:
                    _validate_parameter_value_type(
                        value,
                        parameter_spec.dict_values_type,
                        parameter_spec.label,
                    )

            _validate_param_enums(
                value,
                parameter_spec.dict_values_enums,
                parameter_spec.label,
                inputdata,
                data_model_cdes,
                algorithm_x,
                algorithm_y,
            )


def _validate_parameter_inside_min_max(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
):
    if parameter_spec.min is None and parameter_spec.max is None:
        return

    if parameter_spec.min is not None and parameter_value < parameter_spec.min:
        raise BadUserInput(
            f"Parameter '{parameter_spec.label}' values "
            f"should be greater than {parameter_spec.min} ."
        )

    if parameter_spec.max is not None and parameter_value > parameter_spec.max:
        raise BadUserInput(
            f"Parameter '{parameter_spec.label}' values "
            f"should be at most equal to {parameter_spec.max} ."
        )


def _validate_flags(flags: Dict[str, Any], smpc_enabled: bool, smpc_optional: bool):
    if not flags:
        return

    for flag, value in flags.items():
        if not isinstance(value, bool):
            raise BadUserInput(f"Flag '{flag}' should have a boolean value.")

    available_flags = [f.value for f in AnalysisRequestSystemFlags]
    for flag in flags:
        if flag not in available_flags:
            raise BadUserInput(f"Flag '{flag}' does not exist in the specifications.")

    if AnalysisRequestSystemFlags.SMPC in flags.keys():
        validate_smpc_usage(
            flags[AnalysisRequestSystemFlags.SMPC], smpc_enabled, smpc_optional
        )
