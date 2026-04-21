from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Mapping
from typing import Tuple
from typing import Type

import pandas as pd

from exaflow.algorithms.specifications import PreprocessingStepOrder
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


def get_ordered_preprocessing_items(
    *,
    preprocessing: Dict[str, object] | None,
    preprocessing_step_classes: Mapping[str, Type[PreprocessingStep]],
    preprocessing_step_specs: Mapping[str, PreprocessingStepSpecification]
    | None = None,
) -> List[Tuple[str, object]]:
    """Return preprocessing items sorted by configured step order."""

    ordered_items = list((preprocessing or {}).items())

    def _sort_key(item: Tuple[str, object]) -> Tuple[int, str]:
        step_name, _ = item
        if preprocessing_step_specs and step_name in preprocessing_step_specs:
            return int(preprocessing_step_specs[step_name].order), step_name

        step_cls = preprocessing_step_classes.get(step_name)
        if not step_cls or not hasattr(step_cls, "get_specification"):
            return int(PreprocessingStepOrder.FOURTH), step_name
        try:
            return int(step_cls.get_specification().order), step_name
        except (TypeError, ValueError, NotImplementedError):
            return int(PreprocessingStepOrder.FOURTH), step_name

    return sorted(ordered_items, key=_sort_key)
