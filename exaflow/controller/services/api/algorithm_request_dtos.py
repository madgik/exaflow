from abc import ABC
from enum import Enum
from enum import unique
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict


@unique
class AlgorithmRequestSystemFlags(str, Enum):
    SMPC = "smpc"


class ImmutableBaseModel(BaseModel, ABC):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AlgorithmInputDataDTO(ImmutableBaseModel):
    data_model: str
    datasets: List[str]
    validation_datasets: Optional[List[str]] = None
    filters: Optional[dict] = None
    variables: List[str]


PARAMETERS_TYPE = Dict[str, Any]


class PreprocessingRequestStepDTO(ImmutableBaseModel):
    name: str
    parameters: PARAMETERS_TYPE


class AlgorithmExecutionDTO(ImmutableBaseModel):
    x: Optional[List[str]] = None
    y: Optional[List[str]] = None
    parameters: Optional[PARAMETERS_TYPE] = None


class AlgorithmRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: Optional[str] = None
    inputdata: AlgorithmInputDataDTO
    preprocessing: Optional[List[PreprocessingRequestStepDTO]] = None
    algorithm: AlgorithmExecutionDTO
    flags: Optional[Dict[str, Any]] = None
