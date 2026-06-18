from abc import ABC
from enum import Enum
from enum import unique
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import RootModel

from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterDictValueType
from exaflow.algorithms.specifications import ParameterEnumSpecification
from exaflow.algorithms.specifications import ParameterEnumType
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import ParameterType
from exaflow.algorithms.specifications import PreprocessingOutputSpecification
from exaflow.algorithms.specifications import PreprocessingOutputType
from exaflow.algorithms.specifications import PreprocessingStepSpecification
from exaflow.controller.services.api.analysis_request_dtos import (
    AnalysisRequestSystemFlags,
)
from exaflow.controller.services.specifications import specifications


class ImmutableBaseModel(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)


class InputDataSpecificationDTO(ImmutableBaseModel):
    label: str
    desc: str
    types: List[InputDataType]
    required: bool
    stattypes: Optional[List[InputDataStatType]] = None
    min_count: Optional[int] = None
    max_count: Optional[int] = None


class AnalysisInputDataSpecificationDTO(ImmutableBaseModel):
    data_model: InputDataSpecificationDTO
    datasets: InputDataSpecificationDTO
    filters: InputDataSpecificationDTO
    variables: InputDataSpecificationDTO
    validation_datasets: Optional[InputDataSpecificationDTO] = None


class ParameterEnumSpecificationDTO(ImmutableBaseModel):
    type: ParameterEnumType
    source: List[Any]


class ParameterSpecificationDTO(ImmutableBaseModel):
    label: str
    desc: str
    types: List[ParameterType]
    required: bool
    multiple: bool
    default: Any = None
    enums: Optional[ParameterEnumSpecificationDTO] = None
    dict_keys_enums: Optional[ParameterEnumSpecificationDTO] = None
    dict_values_type: Optional[ParameterDictValueType] = None
    dict_values_enums: Optional[ParameterEnumSpecificationDTO] = None
    min: Optional[float] = None
    max: Optional[float] = None


@unique
class PreprocessingOutputTypeDTO(str, Enum):
    NEW_CATEGORICAL_COLUMN = PreprocessingOutputType.NEW_CATEGORICAL_COLUMN.value


class PreprocessingOutputSpecificationDTO(ImmutableBaseModel):
    type: PreprocessingOutputTypeDTO
    code_parameter: Optional[str] = None


class PreprocessingStepSpecificationDTO(ImmutableBaseModel):
    name: str
    desc: str
    documentation: str
    label: str
    parameters: Optional[Dict[str, ParameterSpecificationDTO]] = None
    output: Optional[PreprocessingOutputSpecificationDTO] = None


class AlgorithmSpecificationDTO(ImmutableBaseModel):
    name: str
    desc: str
    documentation: str
    label: str
    y: InputDataSpecificationDTO
    x: Optional[InputDataSpecificationDTO] = None
    requires_validation_datasets: bool = False
    parameters: Optional[Dict[str, ParameterSpecificationDTO]] = None
    required_preprocessing: List[str]
    flags: Optional[List[str]] = None
    type: AlgorithmType


class AlgorithmSpecificationsDTO(RootModel[List[AlgorithmSpecificationDTO]]):
    pass


class PreprocessingStepSpecificationsDTO(
    RootModel[List[PreprocessingStepSpecificationDTO]]
):
    pass


class AnalysisInputDataSpecificationsDTO(RootModel[AnalysisInputDataSpecificationDTO]):
    pass


def _convert_inputdata_specification_to_dto(self: InputDataSpecification):
    # The only difference of the DTO is that it's stattypes is Optional,
    # due to the fact that the datasets/data_model variables are added.
    return InputDataSpecificationDTO(
        label=self.label,
        desc=self.desc,
        types=self.types,
        stattypes=self.stattypes,
        required=self.required,
        min_count=self.min_count,
        max_count=self.max_count,
    )


def _get_data_model_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Data model of the data.",
        desc="The data model that the algorithm will run on.",
        types=[InputDataType.TEXT],
        required=True,
        stattypes=None,
        min_count=None,
        max_count=1,
    )


def _get_validation_datasets_input_data_specification_dto(required: bool = False):
    return InputDataSpecificationDTO(
        label="Set of data to validate.",
        desc="The set of data to validate the algorithm model on.",
        types=[InputDataType.TEXT],
        required=required,
        stattypes=None,
        min_count=None,
        max_count=None,
    )


def _get_datasets_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Set of data to use.",
        desc="The set of data to run the algorithm on.",
        types=[InputDataType.TEXT],
        required=True,
        stattypes=None,
        min_count=None,
        max_count=None,
    )


def _get_filters_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Filters on the data.",
        desc="Filter rules applied before preprocessing and analysis execution.",
        types=[InputDataType.JSONOBJECT],
        required=False,
        stattypes=None,
        min_count=None,
        max_count=1,
    )


def _get_variables_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Variables.",
        desc="Source variables available to preprocessing and analysis execution.",
        types=[InputDataType.TEXT],
        required=True,
        stattypes=None,
        min_count=1,
        max_count=None,
    )


def _get_analysis_inputdata_specification_dto():
    return AnalysisInputDataSpecificationsDTO(
        root=AnalysisInputDataSpecificationDTO(
            validation_datasets=_get_validation_datasets_input_data_specification_dto(),
            data_model=_get_data_model_input_data_specification_dto(),
            datasets=_get_datasets_input_data_specification_dto(),
            filters=_get_filters_input_data_specification_dto(),
            variables=_get_variables_input_data_specification_dto(),
        )
    )


def _convert_parameter_enum_specification_to_dto(spec: ParameterEnumSpecification):
    return ParameterEnumSpecificationDTO(
        type=spec.type,
        source=spec.source,
    )


def _convert_parameter_specification_to_dto(spec: ParameterSpecification):
    return ParameterSpecificationDTO(
        label=spec.label,
        desc=spec.desc,
        types=spec.types,
        required=spec.required,
        multiple=spec.multiple,
        default=spec.default,
        enums=(
            _convert_parameter_enum_specification_to_dto(spec.enums)
            if spec.enums
            else None
        ),
        dict_keys_enums=(
            _convert_parameter_enum_specification_to_dto(spec.dict_keys_enums)
            if spec.dict_keys_enums
            else None
        ),
        dict_values_type=spec.dict_values_type,
        dict_values_enums=(
            _convert_parameter_enum_specification_to_dto(spec.dict_values_enums)
            if spec.dict_values_enums
            else None
        ),
        min=spec.min,
        max=spec.max,
    )


def _convert_preprocessing_output_specification_to_dto(
    spec: PreprocessingOutputSpecification,
):
    return PreprocessingOutputSpecificationDTO(
        type=PreprocessingOutputTypeDTO(spec.type.value),
        code_parameter=spec.code_parameter,
    )


def _convert_transformer_specification_to_dto(spec: PreprocessingStepSpecification):
    return PreprocessingStepSpecificationDTO(
        name=spec.name,
        desc=spec.desc,
        documentation=spec.documentation,
        label=spec.label,
        parameters=(
            {
                name: _convert_parameter_specification_to_dto(value)
                for name, value in spec.parameters.items()
            }
            if spec.parameters
            else None
        ),
        output=(
            _convert_preprocessing_output_specification_to_dto(spec.output)
            if spec.output
            else None
        ),
    )


def _convert_algorithm_specification_to_dto(
    spec: AlgorithmSpecification,
):
    return AlgorithmSpecificationDTO(
        name=spec.name,
        desc=spec.desc,
        documentation=spec.documentation,
        label=spec.label,
        y=_convert_inputdata_specification_to_dto(spec.y),
        x=_convert_inputdata_specification_to_dto(spec.x) if spec.x else None,
        requires_validation_datasets=spec.requires_validation_datasets,
        parameters=(
            {
                name: _convert_parameter_specification_to_dto(value)
                for name, value in spec.parameters.items()
            }
            if spec.parameters
            else None
        ),
        required_preprocessing=spec.required_preprocessing,
        flags=[AnalysisRequestSystemFlags.SMPC],
        type=spec.type,
    )


def _get_algorithm_specifications_dtos(
    algorithms_specs: List[AlgorithmSpecification],
) -> AlgorithmSpecificationsDTO:
    return AlgorithmSpecificationsDTO(
        root=[
            _convert_algorithm_specification_to_dto(spec) for spec in algorithms_specs
        ]
    )


def _get_preprocessing_step_specifications_dtos(
    preprocessing_steps_specs: List[PreprocessingStepSpecification],
) -> PreprocessingStepSpecificationsDTO:
    return PreprocessingStepSpecificationsDTO(
        root=[
            _convert_transformer_specification_to_dto(spec)
            for spec in preprocessing_steps_specs
        ]
    )


inputdata_specification_dto = _get_analysis_inputdata_specification_dto()


preprocessing_step_specifications_dtos = _get_preprocessing_step_specifications_dtos(
    list(specifications.enabled_preprocessing_steps.values()),
)


algorithm_specifications_dtos = _get_algorithm_specifications_dtos(
    list(specifications.enabled_algorithms.values()),
)
