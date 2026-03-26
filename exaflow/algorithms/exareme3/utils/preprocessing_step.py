import inspect
import json
from abc import ABC
from abc import abstractmethod
from pathlib import Path
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
        inputdata: Inputdata,
        metadata: Dict[str, dict],
        params: Dict[str, object],
    ) -> None:
        self._inputdata = inputdata
        self._metadata = metadata
        self._params = params
        self._validated = False

    @classmethod
    def get_specification(cls) -> PreprocessingStepSpecification:
        transformer_path = Path(inspect.getfile(cls)).resolve()
        transformer_folder = transformer_path.parent

        candidate_paths = [transformer_folder / f"{transformer_path.stem}.json"]
        specification_path = next(
            (path for path in candidate_paths if path.exists()), None
        )
        if specification_path is None:
            expected = ", ".join(str(path) for path in candidate_paths)
            raise FileNotFoundError(
                f"Specification JSON not found for '{cls.__name__}'. Expected one of: {expected}"
            )

        with specification_path.open("r", encoding="utf-8") as fp:
            specification = json.load(fp)

        return PreprocessingStepSpecification.model_validate(specification)

    @classmethod
    def required_input_variables(cls) -> List[str]:
        """
        Additional variables needed from the preprocessing step
        """
        return []

    @abstractmethod
    def validate(self) -> None:
        """Parse and validate configured preprocessing state."""

    @abstractmethod
    def transform_inputdata(
        self,
    ) -> Inputdata:
        """Transform only inputdata-level contract (x/y naming/order)."""

    @abstractmethod
    def transform_metadata(
        self,
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
    ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        """Convenience wrapper for transform_data + transform_metadata."""
        return self.transform_data(data=data), self.transform_metadata()
