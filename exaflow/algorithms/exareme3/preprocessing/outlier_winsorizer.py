from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import pandas as pd

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.preprocessing.metadata import (
    promote_int_variables_to_real,
)
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.federated.statistics.outlier_report import DEFAULT_FOLDS
from exaflow.algorithms.federated.statistics.outlier_report import OutlierBounds
from exaflow.algorithms.federated.statistics.outlier_report import OutlierRule
from exaflow.algorithms.federated.statistics.outlier_report import OutlierStrategy
from exaflow.algorithms.federated.statistics.outlier_report import OutlierTail
from exaflow.algorithms.federated.statistics.outlier_report import WinsorizationHelper
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError


class OutlierWinsorizer(PreprocessingStep):
    """Preprocessing step that clips configured numerical variables locally."""

    def __init__(
        self,
        *,
        params: Dict[str, object],
    ):
        super().__init__(params=params)
        self._strategies = self._dict_param("strategies")
        self._tails = self._dict_param("tails")
        self._folds = self._dict_param("folds")
        self._metadata_bounds: Dict[str, tuple[Optional[float], Optional[float]]] = {}

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        strategy_values = [strategy.value for strategy in OutlierStrategy]
        tail_values = [tail.value for tail in OutlierTail]
        documentation = (
            "Clip numerical outliers using winsorization bounds computed "
            "independently on each worker's local dataset, not from global "
            "statistics.\n\n"
            "Configure one clipping strategy per variable with 'strategies':\n"
            "  - 'gaussian' clips values outside mean +/- fold * sample standard "
            "deviation, computed locally on each worker. Default fold is 3.0.\n"
            "  - 'iqr' clips values outside Q1 - fold * IQR and Q3 + fold * IQR, "
            "where IQR=Q3-Q1, computed locally on each worker. Default fold is 1.5.\n"
            "  - 'mad' clips values outside median +/- fold * normalized MAD, where "
            "normalized MAD=1.4826 * median absolute deviation. Default fold is "
            "3.0. The median and MAD are computed locally on each worker.\n"
            "  - 'quantile' clips values below the fold quantile and above the "
            "1 - fold quantile. Default fold is 0.05. Its fold must be greater "
            "than 0 and less than 0.5. Quantiles are computed locally on each "
            "worker.\n\n"
            "The optional 'tails' setting controls which side is clipped per "
            "variable:\n"
            "  - 'left' clips only values below the lower bound.\n"
            "  - 'right' clips only values above the upper bound.\n"
            "  - 'both' (default) clips both sides for variables without an "
            "explicit tail.\n\n"
            "The optional 'folds' setting overrides the strategy default per "
            "variable:\n"
            "  - 'gaussian', 'iqr', and 'mad' folds must be positive finite "
            "numbers.\n"
            "  - 'quantile' folds must be finite probabilities in (0, 0.5)."
        )
        return specs.PreprocessingStepSpecification(
            name="outlier_winsorizer",
            desc=(
                "Clip numerical outliers per variable using local winsorization bounds."
            ),
            documentation=documentation,
            label="Outlier Winsorizer",
            enabled=True,
            parameters={
                "strategies": specs.ParameterSpecification(
                    label="Strategies",
                    desc="Clipping strategy for each variable.",
                    types=[specs.ParameterType.DICT],
                    required=True,
                    multiple=False,
                    dict_keys_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
                    dict_values_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=strategy_values,
                    ),
                ),
                "tails": specs.ParameterSpecification(
                    label="Tails",
                    desc="Side of the distribution to clip for each variable.",
                    types=[specs.ParameterType.DICT],
                    required=False,
                    multiple=False,
                    dict_keys_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
                    dict_values_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=tail_values,
                    ),
                ),
                "folds": specs.ParameterSpecification(
                    label="Folds",
                    desc="Strategy-specific clipping threshold for each variable.",
                    types=[specs.ParameterType.DICT],
                    required=False,
                    multiple=False,
                    dict_keys_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
                    dict_values_type=specs.ParameterDictValueType.REAL,
                ),
            },
            type=specs.PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
            order=specs.PreprocessingStepOrder.SECOND,
            components=[],
        )

    def validate_params(
        self,
        *,
        inputdata: Inputdata,
        metadata: Dict[str, dict],
    ) -> None:
        self.validate_outlier_configuration(
            strategies=self._strategies,
            folds=self._folds,
            metadata=metadata,
            parameter_label="Outlier winsorizer strategy",
        )

    @staticmethod
    def validate_outlier_configuration(
        *,
        strategies: Dict[str, object],
        folds: Dict[str, object],
        metadata: Dict[str, dict],
        parameter_label: str,
    ) -> None:
        for variable, strategy_value in strategies.items():
            strategy = str(strategy_value)
            if metadata.get(variable, {}).get("is_categorical"):
                raise BadUserInput(
                    f"{parameter_label} '{strategy}' can only be used for "
                    f"numerical variables. Variable '{variable}' is categorical."
                )

            if variable not in folds:
                continue
            fold_value = folds[variable]
            try:
                WinsorizationHelper.resolve_fold(strategy, fold_value)
            except ValueError as exc:
                default = DEFAULT_FOLDS[strategy]
                raise BadUserInput(
                    f"Invalid outlier fold '{fold_value}' for variable '{variable}' "
                    f"with strategy '{strategy}'. Default is {default}."
                ) from exc

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
        transformed_metadata = promote_int_variables_to_real(
            metadata=metadata,
            variables=self._strategies,
        )
        for variable, (lower, upper) in self._metadata_bounds.items():
            if lower is not None:
                transformed_metadata.setdefault(variable, {})["min"] = lower
            if upper is not None:
                transformed_metadata.setdefault(variable, {})["max"] = upper
        return transformed_metadata

    def transform_data(
        self,
        *,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        from exaflow.worker import config as worker_config

        transformed, metadata_bounds = self._clip_data(
            data=data,
            rules=self._make_rules(),
            min_row_count=worker_config.privacy.minimum_row_count,
        )
        self._metadata_bounds = metadata_bounds
        return transformed

    @staticmethod
    def _clip_data(
        *,
        data: pd.DataFrame,
        rules: Dict[str, OutlierRule],
        min_row_count: int,
    ) -> tuple[pd.DataFrame, Dict[str, tuple[Optional[float], Optional[float]]]]:
        if not rules:
            return data, {}

        configured_variables = list(rules)
        missing_columns = sorted(set(configured_variables) - set(data.columns))
        if missing_columns:
            raise BadUserInput(
                "Variables declared in outlier winsorizer preprocessing were not "
                f"found in the runtime data: {missing_columns}."
            )

        for variable in configured_variables:
            data[variable] = pd.to_numeric(data[variable], errors="coerce")
        data.dropna(axis=0, how="any", subset=configured_variables, inplace=True)

        metadata_bounds: Dict[str, tuple[Optional[float], Optional[float]]] = {}
        for variable, rule in rules.items():
            bounds = OutlierWinsorizer._clip_column(
                data=data,
                variable=variable,
                rule=rule,
                min_row_count=min_row_count,
            )
            metadata_bounds[variable] = (bounds.lower, bounds.upper)
        return data, metadata_bounds

    @staticmethod
    def _clip_column(
        *,
        data: pd.DataFrame,
        variable: str,
        rule: OutlierRule,
        min_row_count: int,
    ) -> OutlierBounds:
        num_dtps = int(len(data[variable]))
        if num_dtps < min_row_count:
            raise InsufficientDataError(
                f"Insufficient non-missing data for variable '{rule.variable}': "
                f"{num_dtps} rows; minimum required is {min_row_count}."
            )

        bounds = WinsorizationHelper.compute_bounds(data[variable], rule)
        data[variable] = data[variable].clip(lower=bounds.lower, upper=bounds.upper)
        return bounds

    def _make_rules(self) -> Dict[str, OutlierRule]:
        return WinsorizationHelper.make_rules(
            strategies=self._strategies,
            tails=self._tails,
            folds=self._folds,
        )

    def _dict_param(self, name: str) -> Dict[str, object]:
        value: Optional[object] = self._params.get(name)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise BadUserInput(f"Parameter '{name}' should be a dictionary.")
        return {str(key): str(val) for key, val in value.items()}
