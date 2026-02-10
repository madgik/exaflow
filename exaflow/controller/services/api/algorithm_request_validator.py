import numbers
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataSpecifications
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterEnumSpecification
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import TransformerSpecification
from exaflow.controller.services.api.algorithm_request_dtos import AlgorithmInputDataDTO
from exaflow.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exaflow.controller.services.api.algorithm_request_dtos import (
    AlgorithmRequestSystemFlags,
)
from exaflow.controller.services.api.algorithm_spec_dtos import ParameterEnumType
from exaflow.controller.services.api.algorithm_spec_dtos import ParameterType
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


def validate_algorithm_request(
    algorithm_name: str,
    algorithm_request_dto: AlgorithmRequestDTO,
    algorithms_specs: Dict[str, AlgorithmSpecification],
    transformers_specs: Dict[str, TransformerSpecification],
    worker_landscape_aggregator: WorkerLandscapeAggregator,
    smpc_enabled: bool,
    smpc_optional: bool,
):
    algorithm_specs = _get_algorithm_specs(algorithm_name, algorithms_specs)

    (
        training_datasets,
        validation_datasets,
    ) = worker_landscape_aggregator.get_training_and_validation_datasets(
        algorithm_request_dto.inputdata.data_model
    )
    data_model_cdes = worker_landscape_aggregator.get_cdes(
        algorithm_request_dto.inputdata.data_model
    )
    _validate_algorithm_request_body(
        algorithm_request_dto=algorithm_request_dto,
        algorithm_specs=algorithm_specs,
        transformers_specs=transformers_specs,
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


def _validate_algorithm_request_body(
    algorithm_request_dto: AlgorithmRequestDTO,
    algorithm_specs: AlgorithmSpecification,
    transformers_specs: Dict[str, TransformerSpecification],
    training_datasets: List[str],
    validation_datasets: List[str],
    data_model_cdes: Dict[str, CommonDataElement],
    smpc_enabled: bool,
    smpc_optional: bool,
):
    _validate_inputdata(
        inputdata=algorithm_request_dto.inputdata,
        inputdata_specs=algorithm_specs.inputdata,
        training_datasets=training_datasets,
        algorithm_specification_validation_flag=algorithm_specs.inputdata.validation,
        validation_datasets=validation_datasets,
        data_model_cdes=data_model_cdes,
    )

    _validate_parameters(
        algorithm_request_dto.parameters,
        algorithm_specs.parameters,
        algorithm_request_dto.inputdata,
        data_model_cdes=data_model_cdes,
    )

    _validate_flags(
        flags=algorithm_request_dto.flags,
        smpc_enabled=smpc_enabled,
        smpc_optional=smpc_optional,
    )

    _validate_algorithm_preprocessing(
        algorithm_request_dto=algorithm_request_dto,
        algorithm_name=algorithm_specs.name,
        transformers_specs=transformers_specs,
        data_model_cdes=data_model_cdes,
    )


def _validate_inputdata(
    inputdata: AlgorithmInputDataDTO,
    inputdata_specs: InputDataSpecifications,
    training_datasets: List[str],
    algorithm_specification_validation_flag: Optional[bool],
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
        algorithm_specification_validation_flag=algorithm_specification_validation_flag,
        validation_datasets=validation_datasets,
    )
    _validate_inputdata_filter(inputdata.data_model, inputdata.filters, data_model_cdes)
    _validate_algorithm_inputdatas(inputdata, inputdata_specs, data_model_cdes)


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
    algorithm_specification_validation_flag,
    validation_datasets: List[str],
):
    """
    Validates that the validation dataset values exist.
    """
    if not algorithm_specification_validation_flag and requested_validation_datasets:
        raise BadUserInput(
            "The algorithm does not have a validation flow, but 'validation_datasets' were provided in the 'inputdata'."
        )
    elif algorithm_specification_validation_flag and not requested_validation_datasets:
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
    inputdata: AlgorithmInputDataDTO,
    inputdata_specs: InputDataSpecifications,
    data_model_cdes: Dict[str, CommonDataElement],
):
    if inputdata_specs.x:
        _validate_algorithm_inputdata(inputdata.x, inputdata_specs.x, data_model_cdes)
    if inputdata_specs.y:
        _validate_algorithm_inputdata(inputdata.y, inputdata_specs.y, data_model_cdes)


def _validate_algorithm_inputdata(
    inputdata_values: Optional[List[str]],
    inputdata_spec: InputDataSpecification,
    data_model_cdes: Dict[str, CommonDataElement],
):
    if not inputdata_values and not inputdata_spec:
        return

    if not inputdata_values:
        if inputdata_spec.required:
            raise BadUserInput(
                f"Inputdata '{inputdata_spec.label}' should be provided."
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
        raise BadRequest(f"Inputdata '{inputdata_spec.label}' should be a list.")

    if not inputdata_spec.multiple and len(inputdata_value) > 1:
        raise BadUserInput(
            f"Inputdata '{inputdata_spec.label}' cannot have multiple values."
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
    _validate_inputdata_enumerations(
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
        f"The CDE '{inputdata_value}', of inputdata '{inputdata_specs.label}', "
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
            f"The CDE '{inputdata_value}', of inputdata '{inputdata_specs.label}', "
            f"should be categorical."
        )
    if inputdata_value_metadata.is_categorical and not can_be_nominal:
        raise BadUserInput(
            f"The CDE '{inputdata_value}', of inputdata '{inputdata_specs.label}', "
            f"should NOT be categorical."
        )


def _validate_inputdata_enumerations(
    inputdata_value: str,
    inputdata_specs: InputDataSpecification,
    inputdata_value_metadata: CommonDataElement,
):
    if inputdata_specs.enumslen is not None and inputdata_specs.enumslen != len(
        inputdata_value_metadata.enumerations
    ):
        raise BadUserInput(
            f"The CDE '{inputdata_value}', of inputdata '{inputdata_specs.label}', "
            f"should have {inputdata_specs.enumslen} enumerations."
        )


def _validate_parameters(
    parameters: Optional[Dict[str, Any]],
    parameters_specs: Optional[Dict[str, ParameterSpecification]],
    inputdata: AlgorithmInputDataDTO,
    data_model_cdes: Dict[str, CommonDataElement],
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

        parameter_values = parameters.get(parameter_name)
        if parameter_values:
            _validate_parameter_values(
                parameter_values=parameter_values,
                parameter_spec=parameter_spec,
                inputdata=inputdata,
                data_model_cdes=data_model_cdes,
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
    inputdata: AlgorithmInputDataDTO,
    data_model_cdes: Dict[str, CommonDataElement],
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
        )

        _validate_param_dict_enums(
            parameter_value, parameter_spec, inputdata, data_model_cdes
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


def _validate_param_enums_of_type_input_var_names(
    parameter_value: Any,
    parameter_spec_enums: ParameterEnumSpecification,
    parameter_spec_label: str,
    inputdata: AlgorithmInputDataDTO,
):
    input_var_names_enums = []
    input_var_names_enums.extend(inputdata.y) if inputdata.y else None
    input_var_names_enums.extend(inputdata.x) if inputdata.x else None
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
    inputdata: AlgorithmInputDataDTO,
    data_model_cdes: Dict[str, CommonDataElement],
):
    param_spec_enums_source = parameter_spec_enums.source[
        0
    ]  # Input var CDE enums allows only one source value
    if param_spec_enums_source == "x":
        input_vars = inputdata.x
    elif param_spec_enums_source == "y":
        input_vars = inputdata.y
    else:
        raise NotImplementedError(f"Source should be either 'x' or 'y'.")
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
    inputdata: AlgorithmInputDataDTO,
    data_model_cdes: Dict[str, CommonDataElement],
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
            inputdata,
            data_model_cdes,
        )
    elif parameter_spec_enums.type == ParameterEnumType.FIXED_VAR_CDE_ENUMS:
        _validate_param_enums_of_type_fixed_var_CDE_enums(
            parameter_value, parameter_spec_enums, parameter_spec_label, data_model_cdes
        )
    elif parameter_spec_enums.type == ParameterEnumType.INPUT_VAR_NAMES:
        _validate_param_enums_of_type_input_var_names(
            parameter_value, parameter_spec_enums, parameter_spec_label, inputdata
        )
    else:
        raise NotImplementedError(
            f"Parameter enum type not supported: '{parameter_spec_enums.type}'."
        )


def _validate_param_dict_enums(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
    inputdata: AlgorithmInputDataDTO,
    data_model_cdes: Dict[str, CommonDataElement],
):
    if ParameterType.DICT in parameter_spec.types:
        for key in parameter_value.keys():
            _validate_param_enums(
                key,
                parameter_spec.dict_keys_enums,
                parameter_spec.label,
                inputdata,
                data_model_cdes,
            )

        for value in parameter_value.values():
            _validate_param_enums(
                value,
                parameter_spec.dict_values_enums,
                parameter_spec.label,
                inputdata,
                data_model_cdes,
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

    available_flags = [f.value for f in AlgorithmRequestSystemFlags]
    for flag in flags:
        if flag not in available_flags:
            raise BadUserInput(f"Flag '{flag}' does not exist in the specifications.")

    if AlgorithmRequestSystemFlags.SMPC in flags.keys():
        validate_smpc_usage(
            flags[AlgorithmRequestSystemFlags.SMPC], smpc_enabled, smpc_optional
        )


def _validate_algorithm_preprocessing(
    algorithm_request_dto: AlgorithmRequestDTO,
    algorithm_name: str,
    transformers_specs: Dict[str, TransformerSpecification],
    data_model_cdes: Dict[str, CommonDataElement],
):
    if not algorithm_request_dto.preprocessing:
        return

    for name, params in algorithm_request_dto.preprocessing.items():
        if name not in transformers_specs.keys():
            raise BadUserInput(f"Transformer '{name}' does not exist.")

        compatible_algos = transformers_specs[name].compatible_algorithms
        if compatible_algos and algorithm_name not in compatible_algos:
            raise BadUserInput(
                f"Transformer '{name}' is not available for algorithm '{algorithm_name}'."
            )

        _validate_parameters(
            parameters=params,
            parameters_specs=transformers_specs[name].parameters,
            inputdata=algorithm_request_dto.inputdata,
            data_model_cdes=data_model_cdes,
        )
