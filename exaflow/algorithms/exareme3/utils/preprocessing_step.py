from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd

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

    @abstractmethod
    def validate_params(
        self,
        *,
        inputdata: Inputdata,
        metadata: Dict[str, dict],
    ) -> None:
        """Parse and validate configured preprocessing state."""

    @abstractmethod
    def transform_inputdata_variables(
        self,
        *,
        x: List[str],
        y: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Transform only input variable names (x/y naming/order)."""

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
    ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        """Convenience wrapper for transform_data + transform_metadata."""
        return self.transform_data(data=data), self.transform_metadata(
            metadata=metadata
        )
