from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError


class MissingValueStrategy(str, Enum):
    DROP = "drop"
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class MissingValuesHandler(PreprocessingStep):
    """Preprocessing step for handling missing values."""

    def __init__(
        self,
        *,
        params: Dict[str, object],
    ):
        super().__init__(params=params)

        raw_strategies = self._params.get("strategies") or {}
        self._strategies: Dict[str, str] = {
            str(var): str(value) for var, value in raw_strategies.items()
        }

        raw_fill_values = self._params.get("fill_values") or {}
        self._fill_values: Dict[str, object] = {
            str(var): value for var, value in raw_fill_values.items()
        }

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        allowed_values = [strategy.value for strategy in MissingValueStrategy]
        return specs.PreprocessingStepSpecification(
            name="missing_values_handler",
            desc="Handle missing values per selected variable strategy.",
            label="Missing Values Handler",
            enabled=True,
            parameters={
                "strategies": specs.ParameterSpecification(
                    label="Strategies",
                    desc="Per-variable missing values handling strategy.",
                    types=[specs.ParameterType.DICT],
                    required=True,
                    multiple=False,
                    default=None,
                    enums=None,
                    dict_keys_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
                    dict_values_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=allowed_values,
                    ),
                    min=None,
                    max=None,
                ),
                "fill_values": specs.ParameterSpecification(
                    label="Fill Values",
                    desc="Per-variable fill values used when strategy is 'constant'.",
                    types=[specs.ParameterType.DICT],
                    required=False,
                    multiple=False,
                    default=None,
                    enums=None,
                    dict_keys_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
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
        allowed_values = {strategy.value for strategy in MissingValueStrategy}
        invalid_values = {
            var: strategy
            for var, strategy in self._strategies.items()
            if strategy not in allowed_values
        }
        if invalid_values:
            raise BadUserInput(
                "Invalid per-variable strategy values provided: "
                f"{invalid_values}. Allowed values are: {sorted(allowed_values)}."
            )

        requested_vars = set((inputdata.x or []) + (inputdata.y or []))
        unknown_vars = sorted(set(self._strategies) - requested_vars)
        if unknown_vars:
            raise BadUserInput(
                "Per-variable strategies include variables not present in x/y: "
                f"{unknown_vars}."
            )

        unknown_fill_value_vars = sorted(set(self._fill_values) - set(self._strategies))
        if unknown_fill_value_vars:
            raise BadUserInput(
                "'fill_values' can only be provided for variables declared in "
                f"'strategies'. Unknown variables: {unknown_fill_value_vars}."
            )

        for var, strategy in self._strategies.items():
            if var not in metadata:
                raise BadUserInput(f"Variable '{var}' is missing from metadata.")
            if strategy in (
                MissingValueStrategy.MEAN.value,
                MissingValueStrategy.MEDIAN.value,
            ) and metadata.get(var, {}).get("is_categorical"):
                raise BadUserInput(
                    f"Strategy '{strategy}' can only be used for numerical variables. "
                    f"Variable '{var}' is categorical."
                )

        for var, fill_value in self._fill_values.items():
            if self._strategies[var] != MissingValueStrategy.CONSTANT.value:
                raise BadUserInput(
                    f"'fill_values[{var}]' can only be provided when strategy is "
                    f"'{MissingValueStrategy.CONSTANT.value}'."
                )
            if not _is_supported_fill_value(fill_value):
                raise BadUserInput(
                    f"'fill_values[{var}]' should be a scalar "
                    "(text/int/real/boolean/null)."
                )

        self._validate_categorical_constant_fill_values(metadata=metadata)

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
        return self._apply_column_strategies(
            data=data,
            strategies=self._strategies,
            fill_values=self._fill_values,
        )

    def transform_data_and_metadata(
        self,
        *,
        data: pd.DataFrame,
        metadata: Dict[str, dict],
    ) -> tuple[pd.DataFrame, Dict[str, dict]]:
        # Enforce categorical constant-fill constraints at runtime too.
        self._validate_categorical_constant_fill_values(metadata=metadata)
        return self.transform_data(data=data), self.transform_metadata(
            metadata=metadata
        )

    def _apply_column_strategies(
        self,
        *,
        data: pd.DataFrame,
        strategies: Dict[str, str],
        fill_values: Dict[str, object],
    ) -> pd.DataFrame:
        if not strategies:
            return data.copy()

        missing_columns = sorted(set(strategies) - set(data.columns))
        if missing_columns:
            raise BadUserInput(
                "Variables declared in missing-values preprocessing were not found in "
                f"the runtime data: {missing_columns}."
            )

        transformed = data.copy()
        # Align missing value representation so None/pd.NA are treated consistently.
        transformed = transformed.where(pd.notna(transformed), np.nan)

        drop_columns = [
            column
            for column, strategy in strategies.items()
            if strategy == MissingValueStrategy.DROP.value
        ]
        if drop_columns:
            transformed = transformed.dropna(axis=0, how="any", subset=drop_columns)

        if transformed.empty:
            return transformed

        for column, strategy in strategies.items():
            if strategy == MissingValueStrategy.DROP.value:
                continue

            if (
                strategy
                in {
                    MissingValueStrategy.MEAN.value,
                    MissingValueStrategy.MEDIAN.value,
                    MissingValueStrategy.MOST_FREQUENT.value,
                }
                and transformed[column].isna().all()
            ):
                raise InsufficientDataError(
                    "Insufficient data: variable "
                    f"'{column}' has only missing values on this worker, "
                    f"cannot apply '{strategy}' imputation."
                )

            imputer_kwargs: Dict[str, Any] = {
                "strategy": strategy,
                "keep_empty_features": True,
            }
            if (
                strategy == MissingValueStrategy.CONSTANT.value
                and column in fill_values
            ):
                imputer_kwargs["fill_value"] = fill_values[column]

            try:
                imputer = SimpleImputer(**imputer_kwargs)
                transformed[column] = imputer.fit_transform(
                    transformed[[column]]
                ).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise BadUserInput(
                    "Could not apply missing values strategy "
                    f"'{strategy}' for variable '{column}': {exc}"
                ) from exc

        return transformed

    def _validate_categorical_constant_fill_values(
        self,
        *,
        metadata: Dict[str, dict],
    ) -> None:
        for var, strategy in self._strategies.items():
            if (
                strategy == MissingValueStrategy.CONSTANT.value
                and metadata.get(var, {}).get("is_categorical")
                and var not in self._fill_values
            ):
                raise BadUserInput(
                    f"Categorical variable '{var}' with strategy "
                    f"'{MissingValueStrategy.CONSTANT.value}' requires "
                    f"'fill_values[{var}]' to be one of the variable enum codes."
                )

        for var, fill_value in self._fill_values.items():
            if not metadata.get(var, {}).get("is_categorical"):
                continue
            allowed_enum_codes = _get_allowed_categorical_codes(metadata, var)
            if not isinstance(fill_value, str):
                raise BadUserInput(
                    f"'fill_values[{var}]' should be a text categorical enum code."
                )
            if fill_value not in allowed_enum_codes:
                raise BadUserInput(
                    f"'fill_values[{var}]' should be one of the variable "
                    f"categorical enum codes: {sorted(allowed_enum_codes)}."
                )


def _is_supported_fill_value(value: object) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _get_allowed_categorical_codes(
    metadata: Dict[str, dict],
    variable: str,
) -> set[str]:
    enums = metadata.get(variable, {}).get("enumerations") or {}
    if isinstance(enums, dict):
        return {str(code) for code in enums.keys()}
    return {str(code) for code in enums}
