import inspect
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List

import pandas as pd

from exaflow.algorithms.exareme3.utils.registry import AGGREGATION_CLIENT_PARAMETER_NAME
from exaflow.algorithms.specifications import ComponentType
from exaflow.algorithms.specifications import PreprocessingStepSpecification
from exaflow.algorithms.utils.inputdata_utils import Inputdata


class PreprocessingStep(ABC):
    def __init__(
        self,
        *,
        params: Dict[str, object],
    ) -> None:
        self._params = params

    @classmethod
    @abstractmethod
    def get_specification(cls) -> PreprocessingStepSpecification:
        """Get the preprocessing step specification."""

    @classmethod
    def required_input_variables(cls) -> List[str]:
        """
        Additional variables needed from the preprocessing step
        """
        return []

    @classmethod
    def aggregation_server_required(cls) -> bool:
        return ComponentType.AGGREGATION_SERVER in cls.get_specification().components

    @abstractmethod
    def validate_params(
        self,
        *,
        inputdata: Inputdata,
        metadata: Dict[str, dict],
    ) -> None:
        """Parse and validate configured preprocessing state."""

    @abstractmethod
    def transform_variables(
        self,
        *,
        variables: List[str],
    ) -> List[str]:
        """Transform source variable names."""

    @abstractmethod
    def transform_metadata(
        self,
        *,
        metadata: Dict[str, dict],
    ) -> Dict[str, dict]:
        """Transform metadata used by local UDF execution."""

    @abstractmethod
    def transform_data(
        self,
        *,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform runtime data used by local UDF execution."""

    def transform_data_and_metadata(
        self,
        *,
        data: pd.DataFrame,
        metadata: Dict[str, dict],
        agg_client=None,
    ) -> tuple[pd.DataFrame, Dict[str, dict]]:
        """
        Convenience wrapper for transform_data + transform_metadata.

        Preprocessing steps that declare ComponentType.AGGREGATION_SERVER can
        receive the aggregation client by using this base method and explicitly
        adding the configured aggregation client parameter to transform_data.
        """
        transform_data_kwargs = {"data": data}
        transform_data_signature = inspect.signature(self.transform_data)
        if AGGREGATION_CLIENT_PARAMETER_NAME in transform_data_signature.parameters:
            transform_data_kwargs[AGGREGATION_CLIENT_PARAMETER_NAME] = agg_client
        return self.transform_data(**transform_data_kwargs), self.transform_metadata(
            metadata=metadata
        )
