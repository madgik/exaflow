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
    model_config = ConfigDict(frozen=True)


class AlgorithmInputDataDTO(ImmutableBaseModel):
    data_model: str
    datasets: List[str]
    validation_datasets: Optional[List[str]] = None
    filters: Optional[dict] = None
    y: Optional[List[str]] = None
    x: Optional[List[str]] = None


PARAMETERS_TYPE = Dict[str, Any]


class AlgorithmRequestDTO(BaseModel):
    request_id: Optional[str] = None
    inputdata: AlgorithmInputDataDTO
    parameters: Optional[PARAMETERS_TYPE] = None
    flags: Optional[Dict[str, Any]] = None
    preprocessing: Optional[Dict[str, PARAMETERS_TYPE]] = None
