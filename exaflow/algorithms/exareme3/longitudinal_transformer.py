from __future__ import annotations

from copy import deepcopy
from typing import Dict
from typing import Iterable
from typing import List

import pandas as pd

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.pandas_utils import convert_to_pandas_dataframe
from exaflow.column_names import DATASET_COL
from exaflow.column_names import SUBJECT_ID_COL
from exaflow.column_names import VISIT_ID_COL
from exaflow.worker_communication import BadUserInput

STRATEGY_FIRST = "first"
STRATEGY_SECOND = "second"
STRATEGY_DIFF = "diff"
VALID_STRATEGIES = {STRATEGY_FIRST, STRATEGY_SECOND, STRATEGY_DIFF}

DIFF_SUFFIX = "_diff"
VISIT1_VALUE_SUFFIX = "_v1"
VISIT2_VALUE_SUFFIX = "_v2"


class LongitudinalTransformer(PreprocessingStep):
    def __init__(
        self,
        *,
        params: Dict[str, object],
    ):
        super().__init__(params=params)
        visit1 = self._params.get("visit1")
        visit2 = self._params.get("visit2")
        strategies = self._params.get("strategies", {})

        self._visit1: str = str(visit1) if visit1 is not None else ""
        self._visit2: str = str(visit2) if visit2 is not None else ""
        self._strategies: Dict[str, str] = (
            dict(strategies) if isinstance(strategies, dict) else {}
        )

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name="longitudinal_transformer",
            desc="Longitudinal transformation between two visits.",
            label="Longitudinal Transformer",
            enabled=True,
            parameters={
                "visit1": specs.ParameterSpecification(
                    label="1st Visit",
                    desc="The earlier visit identifier.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default=None,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.FIXED_VAR_CDE_ENUMS,
                        source=[VISIT_ID_COL],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
                "visit2": specs.ParameterSpecification(
                    label="2nd Visit",
                    desc="The later visit identifier.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default=None,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.FIXED_VAR_CDE_ENUMS,
                        source=[VISIT_ID_COL],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
                "strategies": specs.ParameterSpecification(
                    label="Strategies",
                    desc="Select a strategy for each variable.",
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
                        source=[STRATEGY_DIFF, STRATEGY_FIRST, STRATEGY_SECOND],
                    ),
                    min=None,
                    max=None,
                ),
            },
            compatible_algorithms=[
                "anova_oneway",
                "anova_twoway",
                "linear_regression",
                "linear_regression_cv",
                "logistic_regression",
                "logistic_regression_cv",
                "naive_bayes_gaussian_cv",
                "naive_bayes_categorical_cv",
            ],
            type=specs.PreprocessingStep.EXAREME3_PREPROCESSING_STEP,
            components=[],
        )

    @classmethod
    def required_input_variables(cls) -> List[str]:
        return [DATASET_COL, SUBJECT_ID_COL, VISIT_ID_COL]

    def validate_params(
        self,
        *,
        inputdata: Inputdata,
        metadata: Dict[str, dict],
    ) -> None:
        if not self._visit1 or not self._visit2:
            raise BadUserInput("Both 'visit1' and 'visit2' parameters are required.")
        if self._visit1 == self._visit2:
            raise BadUserInput("'visit1' and 'visit2' must be different.")
        if not isinstance(self._params.get("strategies", {}), dict):
            raise BadUserInput("'strategies' must be a dictionary.")

        raw_x = list(inputdata.x or [])
        raw_y = list(inputdata.y or [])
        requested_vars = set(raw_x + raw_y)
        provided_vars = set(self._strategies.keys())

        missing = sorted(requested_vars - provided_vars)
        extra = sorted(provided_vars - requested_vars)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing: {missing}")
            if extra:
                details.append(f"extra: {extra}")
            raise BadUserInput(
                "A strategy must be provided exactly for the variables in x and y "
                f"({' ; '.join(details)})."
            )

        invalid_values = {
            name: value
            for name, value in self._strategies.items()
            if value not in VALID_STRATEGIES
        }
        if invalid_values:
            raise BadUserInput(
                "Invalid strategy values provided: "
                f"{invalid_values}. Allowed values are: {sorted(VALID_STRATEGIES)}."
            )

        self._validate_diff_not_nominal(
            strategies=self._strategies,
            metadata=metadata,
        )

    def transform_inputdata(
        self,
        *,
        inputdata: Inputdata,
    ) -> Inputdata:
        raw_x = list(inputdata.x or [])
        raw_y = list(inputdata.y or [])
        transformed_x = self._build_transformed_variable_names(raw_x)
        transformed_y = self._build_transformed_variable_names(raw_y)
        return inputdata.model_copy(update={"x": transformed_x, "y": transformed_y})

    def transform_metadata(
        self,
        *,
        metadata: Dict[str, dict],
    ) -> Dict[str, dict]:
        transformed_metadata = deepcopy(metadata)
        for varname, strategy in self._strategies.items():
            if strategy == STRATEGY_DIFF:
                source_metadata = transformed_metadata.pop(varname, None)
                if source_metadata is not None:
                    transformed_metadata[_output_name(varname, strategy)] = (
                        source_metadata
                    )
        return transformed_metadata

    def transform_data(
        self,
        *,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed_variables = self._build_transformed_variable_names(
            list(self._strategies.keys())
        )
        df = convert_to_pandas_dataframe(data)
        missing_columns = self._required_longitudinal_columns() - set(df.columns)
        if missing_columns:
            raise BadUserInput(
                "Missing required columns for longitudinal transformation: "
                f"{sorted(missing_columns)}"
            )

        df = df[df[VISIT_ID_COL].isin([self._visit1, self._visit2])]
        key_cols = [SUBJECT_ID_COL]
        if DATASET_COL in df.columns:
            key_cols.append(DATASET_COL)

        left = df[df[VISIT_ID_COL] == self._visit1]
        right = df[df[VISIT_ID_COL] == self._visit2]
        merged = left.merge(
            right,
            on=key_cols,
            suffixes=(VISIT1_VALUE_SUFFIX, VISIT2_VALUE_SUFFIX),
            how="inner",
        )
        result = merged[key_cols].copy()

        strategy_dispatch = {
            STRATEGY_FIRST: lambda series_v1, series_v2: series_v1,
            STRATEGY_SECOND: lambda series_v1, series_v2: series_v2,
            STRATEGY_DIFF: lambda series_v1, series_v2: series_v2 - series_v1,
        }
        for varname, strategy in self._strategies.items():
            value_visit1 = merged[f"{varname}{VISIT1_VALUE_SUFFIX}"]
            value_visit2 = merged[f"{varname}{VISIT2_VALUE_SUFFIX}"]
            result[_output_name(varname, strategy)] = strategy_dispatch[strategy](
                value_visit1, value_visit2
            )

        desired_columns = _deduplicate_preserve_order(key_cols + transformed_variables)
        return result[[col for col in desired_columns if col in result.columns]]

    def _required_longitudinal_columns(self) -> set[str]:
        return set(list(self._strategies.keys()) + [SUBJECT_ID_COL, VISIT_ID_COL])

    def _validate_diff_not_nominal(
        self, *, strategies: Dict[str, str], metadata: Dict[str, dict]
    ) -> None:
        for name, strategy in strategies.items():
            if strategy == STRATEGY_DIFF and metadata.get(name, {}).get(
                "is_categorical"
            ):
                raise BadUserInput(
                    f"Cannot take the difference for the nominal variable '{name}'."
                )

    def _build_transformed_variable_names(self, variables: List[str]) -> List[str]:
        return [
            _output_name(name, self._strategies.get(name, STRATEGY_FIRST))
            for name in variables
        ]


def _output_name(varname: str, strategy: str) -> str:
    return f"{varname}{DIFF_SUFFIX}" if strategy == STRATEGY_DIFF else varname


def _deduplicate_preserve_order(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))
