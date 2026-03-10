from abc import ABC
from enum import Enum
from enum import unique
from ipaddress import IPv4Address
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from exaflow import DType


class RequestIDNotFound(Exception):
    """Exception raised while checking the presence of request_id in task's arguments.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self):
        self.message = "Request id is missing from task's arguments."
        super().__init__(self.message)


class DataModelUnavailable(Exception):
    """
    Exception raised when a data model is not available in the WORKER db.

    Attributes:
        worker_id -- the worker id that threw the exception
        data_model --  the unavailable data model
        message -- explanation of the error
    """

    def __init__(self, worker_id: str, data_model: str):
        self.worker_id = worker_id
        self.data_model = data_model
        self.message = f"Data model '{self.data_model}' is not available in worker: '{self.worker_id}'."


class DatasetUnavailable(Exception):
    """
    Exception raised when a dataset is not available in the WORKER db.

    Attributes:
        worker_id -- the worker id that threw the exception
        dataset --  the unavailable dataset
        message -- explanation of the error
    """

    def __init__(self, worker_id: str, dataset: str):
        self.worker_id = worker_id
        self.dataset = dataset
        self.message = (
            f"Dataset '{self.dataset}' is not available in worker: '{self.worker_id}'."
        )


class InsufficientDataError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class BadUserInput(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


# ~~~~~~~~~~~~~~~~~~~~ Enums ~~~~~~~~~~~~~~~~~~~~ #


@unique
class WorkerRole(str, Enum):
    GLOBALWORKER = "GLOBALWORKER"
    LOCALWORKER = "LOCALWORKER"


# ~~~~~~~~~~~~~~~~~~~ DTOs ~~~~~~~~~~~~~~~~~~~~~~ #


class ImmutableBaseModel(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)


class WorkerInfo(BaseModel):
    id: str
    role: WorkerRole
    ip: IPv4Address
    port: int
    data_folder: Optional[str] = None
    auto_load_data: bool = False

    @property
    def socket_addr(self) -> str:
        return f"{self.ip}:{self.port}"


class DatasetProperties(ImmutableBaseModel):
    tags: List
    properties: Dict[str, List[str]]

    @field_validator("properties")
    @classmethod
    def validate_variables(cls, properties: Dict[str, List[str]]):
        variables = properties.get("variables")
        if not isinstance(variables, list):
            raise ValueError("Dataset properties must include a 'variables' list.")
        if not all(isinstance(variable, str) for variable in variables):
            raise ValueError("Dataset variables must be strings.")
        return properties

    @property
    def variables(self) -> List[str]:
        return self.properties["variables"]


class DataModelAttributes(ImmutableBaseModel):
    tags: List
    properties: Dict


class DatasetInfo(ImmutableBaseModel):
    code: str
    label: str
    variables: List[str] = Field(default_factory=list)


class DatasetsInfoPerDataModel(ImmutableBaseModel):
    datasets_info_per_data_model: Dict[str, List[DatasetInfo]]


class CommonDataElement(ImmutableBaseModel):
    code: str
    label: str
    sql_type: str
    is_categorical: bool
    enumerations: Optional[Dict[str, str]] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def __eq__(self, other):
        return (
            isinstance(other, CommonDataElement)
            and self.code == other.code
            and self.label == other.label
            and self.sql_type == other.sql_type
            and self.is_categorical == other.is_categorical
            and self.enumerations == other.enumerations
            and self.max == other.max
            and self.min == other.min
        )


class CommonDataElements(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    values: Dict[str, CommonDataElement]

    def __eq__(self, other):
        """
        We are overriding the equals function to check that the two cdes have identical fields except one edge case.
        The edge case is that the two comparing cdes can only contain a difference in the field of enumerations in
        the cde with code 'dataset' and still be considered compatible.
        """
        if set(self.values.keys()) != set(other.values.keys()):
            return False
        for cde_code in self.values.keys():
            cde1 = self.values[cde_code]
            cde2 = other.values[cde_code]
            if not cde1 == cde2 and not self._are_equal_dataset_cdes(cde1, cde2):
                return False
        return True

    def _are_equal_dataset_cdes(
        self, cde1: CommonDataElement, cde2: CommonDataElement
    ) -> bool:
        if cde1.code != "dataset" or cde2.code != "dataset":
            return False

        if (
            cde1.label != cde2.label
            or cde1.sql_type != cde2.sql_type
            or cde1.is_categorical != cde2.is_categorical
            or cde1.max != cde2.max
            or cde1.min != cde2.min
        ):
            return False

        return True


# ~~~~~~~~~~~~~~~~~~~ Table Data DTOs ~~~~~~~~~~~~~~~~~~~~~~ #


class ColumnData(ImmutableBaseModel):
    name: str
    data: List[Any]
    type: DType

    @field_validator("type")
    @classmethod
    def validate_type(cls, tp):
        if cls.__name__ == "ColumnData":
            raise TypeError(
                "ColumnData should not be instantiated. "
                "Use ColumnDataInt, ColumnDataStr, ColumnDataFloat, ColumnDataJSON  or ColumnDataBinary instead."
            )
        column_type = cls.model_fields["type"].default
        if tp != column_type:
            raise ValueError(
                f"Objects of type {cls.__name__} have a fixed type {column_type}, "
                f"you cannot use {tp} in the constructor."
            )
        return tp


class ColumnDataInt(ColumnData):
    data: List[Union[None, int]]
    type: DType = DType.INT


class ColumnDataStr(ColumnData):
    data: List[Union[None, str]]
    type: DType = DType.STR


class ColumnDataFloat(ColumnData):
    data: List[Union[None, float]]
    type: DType = DType.FLOAT


class TabularDataResult(ImmutableBaseModel):
    title: str
    columns: List[Union[ColumnDataInt, ColumnDataStr, ColumnDataFloat]]
