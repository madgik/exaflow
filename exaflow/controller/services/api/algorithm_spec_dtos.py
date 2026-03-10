from abc import ABC
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
from exaflow.algorithms.specifications import InputDataSpecifications
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.algorithms.specifications import ParameterEnumSpecification
from exaflow.algorithms.specifications import ParameterEnumType
from exaflow.algorithms.specifications import ParameterSpecification
from exaflow.algorithms.specifications import ParameterType
from exaflow.algorithms.specifications import TransformerSpecification
from exaflow.controller.services.api.algorithm_request_dtos import (
    AlgorithmRequestSystemFlags,
)
from exaflow.controller.services.specifications import specifications


class ImmutableBaseModel(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)


class InputDataSpecificationDTO(ImmutableBaseModel):
    label: str
    desc: str
    types: List[InputDataType]
    required: bool
    multiple: bool
    stattypes: Optional[List[InputDataStatType]] = None
    enumslen: Optional[int] = None


class InputDataSpecificationsDTO(ImmutableBaseModel):
    data_model: InputDataSpecificationDTO
    datasets: InputDataSpecificationDTO
    filter: InputDataSpecificationDTO
    y: InputDataSpecificationDTO
    x: Optional[InputDataSpecificationDTO] = None
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
    dict_values_enums: Optional[ParameterEnumSpecificationDTO] = None
    min: Optional[float] = None
    max: Optional[float] = None


class TransformerSpecificationDTO(ImmutableBaseModel):
    name: str
    desc: str
    label: str
    parameters: Optional[Dict[str, ParameterSpecificationDTO]] = None


class AlgorithmSpecificationDTO(ImmutableBaseModel):
    name: str
    desc: str
    label: str
    inputdata: InputDataSpecificationsDTO
    parameters: Optional[Dict[str, ParameterSpecificationDTO]] = None
    preprocessing: Optional[List[TransformerSpecificationDTO]] = None
    flags: Optional[List[str]] = None
    type: AlgorithmType


class AlgorithmSpecificationsDTO(RootModel[List[AlgorithmSpecificationDTO]]):
    pass


class TransformerSpecificationsDTO(RootModel[List[TransformerSpecificationDTO]]):
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
        multiple=self.multiple,
        enumslen=self.enumslen,
    )


def _get_data_model_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Data model of the data.",
        desc="The data model that the algorithm will run on.",
        types=[InputDataType.TEXT],
        required=True,
        multiple=False,
        stattypes=None,
        enumslen=None,
    )


def _get_validation_datasets_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Set of data to validate.",
        desc="The set of data to validate the algorithm model on.",
        types=[InputDataType.TEXT],
        required=True,
        multiple=True,
        stattypes=None,
        enumslen=None,
    )


def _get_datasets_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="Set of data to use.",
        desc="The set of data to run the algorithm on.",
        types=[InputDataType.TEXT],
        required=True,
        multiple=True,
        stattypes=None,
        enumslen=None,
    )


def _get_filters_input_data_specification_dto():
    return InputDataSpecificationDTO(
        label="filter on the data.",
        desc="Features used in my algorithm.",
        types=[InputDataType.JSONOBJECT],
        required=False,
        multiple=False,
        stattypes=None,
        enumslen=None,
    )


def _convert_inputdata_specifications_to_dto(spec: InputDataSpecifications):
    # In the DTO the datasets, data_model and filter parameters are added from the engine.
    # These parameters are not added by the algorithm developer.
    y = _convert_inputdata_specification_to_dto(spec.y)
    x = _convert_inputdata_specification_to_dto(spec.x) if spec.x else None
    validation_datasets_dto = (
        _get_validation_datasets_input_data_specification_dto()
        if spec.validation
        else None
    )
    return InputDataSpecificationsDTO(
        y=y,
        x=x,
        validation_datasets=validation_datasets_dto,
        data_model=_get_data_model_input_data_specification_dto(),
        datasets=_get_datasets_input_data_specification_dto(),
        filter=_get_filters_input_data_specification_dto(),
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
        dict_values_enums=(
            _convert_parameter_enum_specification_to_dto(spec.dict_values_enums)
            if spec.dict_values_enums
            else None
        ),
        min=spec.min,
        max=spec.max,
    )


def _convert_transformer_specification_to_dto(spec: TransformerSpecification):
    return TransformerSpecificationDTO(
        name=spec.name,
        desc=spec.desc,
        label=spec.label,
        parameters=(
            {
                name: _convert_parameter_specification_to_dto(value)
                for name, value in spec.parameters.items()
            }
            if spec.parameters
            else None
        ),
    )


def _get_algorithm_compatible_transformers(
    algo_name: str, transformers: List[TransformerSpecification]
) -> List[TransformerSpecification]:
    compatible_transformers = []
    for transformer in transformers:
        if (
            not transformer.compatible_algorithms
            or algo_name in transformer.compatible_algorithms
        ):
            compatible_transformers.append(transformer)
    return compatible_transformers


def _convert_algorithm_specification_to_dto(
    spec: AlgorithmSpecification, transformers: List[TransformerSpecification]
):
    """
    Converting to a DTO has the following additions:
    1) The preprocessing specifications are added from the transformers that are compatible with the specific algorithm.
    2) The system specific flags are added.
    """
    return AlgorithmSpecificationDTO(
        name=spec.name,
        desc=spec.desc,
        label=spec.label,
        inputdata=_convert_inputdata_specifications_to_dto(spec.inputdata),
        parameters=(
            {
                name: _convert_parameter_specification_to_dto(value)
                for name, value in spec.parameters.items()
            }
            if spec.parameters
            else None
        ),
        preprocessing=[
            _convert_transformer_specification_to_dto(spec)
            for spec in _get_algorithm_compatible_transformers(spec.name, transformers)
        ],
        flags=[AlgorithmRequestSystemFlags.SMPC],
        type=spec.type,
    )


def _get_algorithm_specifications_dtos(
    algorithms_specs: List[AlgorithmSpecification],
    transformers_specs: List[TransformerSpecification],
) -> AlgorithmSpecificationsDTO:
    return AlgorithmSpecificationsDTO(
        root=[
            _convert_algorithm_specification_to_dto(spec, transformers_specs)
            for spec in algorithms_specs
        ]
    )


algorithm_specifications_dtos = _get_algorithm_specifications_dtos(
    list(specifications.enabled_algorithms.values()),
    list(specifications.enabled_transformers.values()),
)
