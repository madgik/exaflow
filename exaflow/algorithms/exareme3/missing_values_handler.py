from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Dict
from typing import List

import pandas as pd

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput


class MissingValueStrategy(str, Enum):
    DROP = "drop"


class MissingValuesHandler(PreprocessingStep):
    """Preprocessing step for handling missing values."""

    def __init__(
        self,
        *,
        params: Dict[str, object],
    ):
        super().__init__(params=params)
        strategy = self._params.get("strategy")
        self._strategy = str(strategy) if strategy is not None else ""

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name="missing_values_handler",
            desc="Handle missing values, based on the strategy provided.",
            label="Missing Values Handler",
            enabled=True,
            parameters={
                "strategy": specs.ParameterSpecification(
                    label="Strategy",
                    desc="Missing values handling strategy.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default=MissingValueStrategy.DROP.value,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=[MissingValueStrategy.DROP.value],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
            },
            type=specs.PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
            components=[],
        )

    def validate_params(
        self,
        *,
        inputdata: Inputdata,
        metadata: Dict[str, dict],
    ) -> None:
        allowed_values = [strategy.value for strategy in MissingValueStrategy]
        if self._strategy not in allowed_values:
            raise BadUserInput(
                f"Invalid strategy '{self._strategy}'. "
                f"Allowed values are: {sorted(allowed_values)}."
            )

    def transform_inputdata_variables(
        self,
        *,
        x: List[str],
        y: List[str],
    ) -> tuple[List[str], List[str]]:
        return list(x), list(y)

    def transform_metadata(
        self,
        *,
        metadata: Dict[str, dict],
    ) -> Dict[str, dict]:
        return deepcopy(metadata)

    def transform_data(
        self,
        *,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._strategy == MissingValueStrategy.DROP.value:
            return data.dropna(axis=0, how="any")
        return data
