from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Dict
from typing import Iterable
from typing import List

import pandas as pd

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.column_names import DATASET_COL
from exaflow.column_names import SUBJECT_ID_COL
from exaflow.column_names import VISIT_ID_COL
from exaflow.worker_communication import BadUserInput


class LongitudinalStrategy(str, Enum):
    FIRST = "first"
    SECOND = "second"
    DIFF = "diff"


VISIT1_VALUE_SUFFIX = "_v1"
VISIT2_VALUE_SUFFIX = "_v2"


class LongitudinalTransformer(PreprocessingStep):
    """Federated preprocessing step that aligns and transforms two visits per subject.

    This step expects two visit identifiers (`visit1`, `visit2`) and a per-variable
    strategy map (`strategies`) for all requested inputdata variables.

    Per worker, it:
    1. Filters rows to the selected visits.
    2. Matches records between visits on `subjectid` and `dataset`.
    3. Applies one strategy per variable:
       - `first`: keep value from `visit1`
       - `second`: keep value from `visit2`
       - `diff`: compute `visit2 - visit1` (numeric variables only)

    Output contains only matched subjects and preserves key columns
    (`subjectid`, plus `dataset`). For the `diff` strategy, variable values
    are transformed to `visit2 - visit1`, but the original variable code is
    preserved.

    Validation enforces:
    - `visit1` and `visit2` are both provided and distinct.
    - `strategies` contains exactly the inputdata variables.
    - strategy values are in `{first, second, diff}`.
    - `diff` is not used for categorical variables.
    """

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
            documentation=(
                "Transform selected variables between two visits using a "
                "per-variable longitudinal strategy.\n\n"
                "The 'visit1' setting selects the earlier visit identifier.\n\n"
                "The 'visit2' setting selects the later visit identifier.\n\n"
                "Configure one transformation strategy per variable with "
                "'strategies':\n"
                "  - 'diff' subtracts the first-visit value from the second-visit value.\n"
                "  - 'first' keeps the first-visit value.\n"
                "  - 'second' keeps the second-visit value."
            ),
            label="Longitudinal Transformer",
            enabled=True,
            parameters={
                "visit1": specs.ParameterSpecification(
                    label="1st Visit",
                    desc="The earlier visit identifier.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.FIXED_VAR_CDE_ENUMS,
                        source=[VISIT_ID_COL],
                    ),
                ),
                "visit2": specs.ParameterSpecification(
                    label="2nd Visit",
                    desc="The later visit identifier.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.FIXED_VAR_CDE_ENUMS,
                        source=[VISIT_ID_COL],
                    ),
                ),
                "strategies": specs.ParameterSpecification(
                    label="Strategies",
                    desc="Longitudinal transformation strategy for each variable.",
                    types=[specs.ParameterType.DICT],
                    required=True,
                    multiple=False,
                    dict_keys_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["variables"],
                    ),
                    dict_values_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=[
                            LongitudinalStrategy.DIFF.value,
                            LongitudinalStrategy.FIRST.value,
                            LongitudinalStrategy.SECOND.value,
                        ],
                    ),
                ),
            },
            type=specs.PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
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

        requested_vars = set(inputdata.variables)
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
                "A strategy must be provided exactly for the inputdata variables "
                f"({' ; '.join(details)})."
            )

        valid_strategies = [strategy.value for strategy in LongitudinalStrategy]
        invalid_values = {
            name: value
            for name, value in self._strategies.items()
            if value not in valid_strategies
        }
        if invalid_values:
            raise BadUserInput(
                "Invalid strategy values provided: "
                f"{invalid_values}. Allowed values are: {sorted(valid_strategies)}."
            )

        self._validate_diff_not_nominal(
            strategies=self._strategies,
            metadata=metadata,
        )

    def transform_variables(
        self,
        *,
        variables: List[str],
    ) -> List[str]:
        return self._build_transformed_variable_names(variables)

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
        transformed_variables = self._build_transformed_variable_names(
            list(self._strategies.keys())
        )
        missing_columns = self._required_longitudinal_columns() - set(data.columns)
        if missing_columns:
            raise BadUserInput(
                "Missing required columns for longitudinal transformation: "
                f"{sorted(missing_columns)}"
            )

        data = data[data[VISIT_ID_COL].isin([self._visit1, self._visit2])]
        key_cols = [SUBJECT_ID_COL]
        if DATASET_COL in data.columns:
            key_cols.append(DATASET_COL)

        left = data[data[VISIT_ID_COL] == self._visit1]
        right = data[data[VISIT_ID_COL] == self._visit2]
        merged = left.merge(
            right,
            on=key_cols,
            suffixes=(VISIT1_VALUE_SUFFIX, VISIT2_VALUE_SUFFIX),
            how="inner",
        )
        result = merged[key_cols].copy()

        strategy_dispatch = {
            LongitudinalStrategy.FIRST.value: lambda series_v1, series_v2: series_v1,
            LongitudinalStrategy.SECOND.value: lambda series_v1, series_v2: series_v2,
            LongitudinalStrategy.DIFF.value: lambda series_v1, series_v2: (
                series_v2 - series_v1
            ),
        }
        for varname, strategy in self._strategies.items():
            value_visit1 = merged[f"{varname}{VISIT1_VALUE_SUFFIX}"]
            value_visit2 = merged[f"{varname}{VISIT2_VALUE_SUFFIX}"]
            result[varname] = strategy_dispatch[strategy](value_visit1, value_visit2)

        desired_columns = _deduplicate_preserve_order(key_cols + transformed_variables)
        return result[[col for col in desired_columns if col in result.columns]]

    def _required_longitudinal_columns(self) -> set[str]:
        return set(list(self._strategies.keys()) + [SUBJECT_ID_COL, VISIT_ID_COL])

    def _validate_diff_not_nominal(
        self, *, strategies: Dict[str, str], metadata: Dict[str, dict]
    ) -> None:
        for name, strategy in strategies.items():
            if strategy == LongitudinalStrategy.DIFF.value and metadata.get(
                name, {}
            ).get("is_categorical"):
                raise BadUserInput(
                    f"Cannot take the difference for the nominal variable '{name}'."
                )

    def _build_transformed_variable_names(self, variables: List[str]) -> List[str]:
        return list(variables)


def _deduplicate_preserve_order(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))
