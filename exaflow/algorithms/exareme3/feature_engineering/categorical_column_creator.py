from __future__ import annotations

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List

import pandas as pd

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.output_specifications import resolve_output_metadata
from exaflow.worker_communication import BadUserInput

FILTER_RULES_STRATEGY = "filter_rules"


class CategoricalColumnCreator(PreprocessingStep):
    def __init__(
        self,
        *,
        params: Dict[str, object],
    ):
        super().__init__(params=params)
        self._code = next(
            iter(resolve_output_metadata(self.get_specification(), self._params, {}))
        )
        self._strategy = str(self._params.get("strategy", ""))
        rules = self._params.get("rules", {})
        self._rules: Dict[str, dict] = dict(rules) if isinstance(rules, dict) else {}
        default_enumeration = self._params.get("default_enumeration")
        self._default_enumeration = (
            str(default_enumeration) if default_enumeration is not None else None
        )

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name="categorical_column_creator",
            desc="Creates a new categorical column based on filter rules.",
            documentation=(
                "Creates a nominal text variable by evaluating filter rules "
                "against the requested source variables. Each rule key becomes "
                "an enumeration. Rows matching exactly one rule receive that "
                "enumeration; rows matching none receive the default enumeration "
                "when provided, otherwise null. Rows matching multiple rules are "
                "rejected. Supply one name in new_column_names for the created column."
            ),
            label="Categorical Column Creator",
            enabled=True,
            parameters={
                "new_column_names": specs.ParameterSpecification(
                    label="New column names",
                    desc="Names assigned to the new categorical columns.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=True,
                ),
                "strategy": specs.ParameterSpecification(
                    label="Creation strategy",
                    desc="Strategy used to create the new categorical column.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=[FILTER_RULES_STRATEGY],
                    ),
                    default=FILTER_RULES_STRATEGY,
                ),
                "rules": specs.ParameterSpecification(
                    label="Enumeration filters",
                    desc="Dictionary where each key is an enumeration value and each value is a filter that creates it.",
                    types=[specs.ParameterType.DICT],
                    required=True,
                    multiple=False,
                    dict_values_type=specs.ParameterDictValueType.FILTER,
                ),
                "default_enumeration": specs.ParameterSpecification(
                    label="Default enumeration",
                    desc="Enumeration assigned when no rule matches.",
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                ),
            },
            outputs=specs.PreprocessingOutputsSpecification(
                new_column_names_param="new_column_names",
                columns=[
                    specs.PreprocessingOutputColumnSpecification(
                        type=specs.InputDataType.TEXT,
                        stattype=specs.InputDataStatType.NOMINAL,
                        enumerations=specs.PreprocessingOutputEnumerationsSpecification(
                            source_parameter_name="rules",
                            source_type=specs.PreprocessingOutputEnumerationSource.DICT_KEYS,
                            additional_value_parameter_name="default_enumeration",
                        ),
                    )
                ],
            ),
            type=specs.PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
            components=[],
        )

    def validate_params(
        self,
        *,
        inputdata: Inputdata,
        metadata: Dict[str, dict],
    ) -> None:
        if self._code in metadata:
            raise BadUserInput(
                f"Preprocessing step 'categorical_column_creator' cannot create CDE '{self._code}' because it already exists."
            )
        if self._strategy != FILTER_RULES_STRATEGY:
            raise BadUserInput(
                f"'strategy' should be '{FILTER_RULES_STRATEGY}'. Value provided: '{self._strategy}'."
            )
        if not self._rules:
            raise BadUserInput("'rules' parameter should not be blank.")
        invalid_enumerations = [
            key for key in self._rules.keys() if not isinstance(key, str) or not key
        ]
        if invalid_enumerations:
            raise BadUserInput(
                f"Rule enumeration values should be non-empty strings: {invalid_enumerations}."
            )

    def transform_variables(
        self,
        *,
        variables: List[str],
    ) -> List[str]:
        return list(variables) + [self._code]

    def transform_metadata(
        self,
        *,
        metadata: Dict[str, dict],
    ) -> Dict[str, dict]:
        transformed_metadata = deepcopy(metadata)
        transformed_metadata.update(
            resolve_output_metadata(self.get_specification(), self._params, metadata)
        )
        return transformed_metadata

    def transform_data(
        self,
        *,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        match_count = pd.Series(0, index=data.index)
        output = pd.Series(
            [self._default_enumeration] * len(data),
            index=data.index,
            dtype=object,
        )

        for enumeration, filter_ in self._rules.items():
            mask = _evaluate_filter(data, filter_)
            match_count = match_count + mask.astype(int)
            output.loc[mask] = enumeration

        if (match_count > 1).any():
            raise BadUserInput(
                "Rows matched multiple categorical_column_creator rules."
            )

        data[self._code] = output
        return data


def _evaluate_filter(data: pd.DataFrame, filter_: dict) -> pd.Series:
    if "condition" in filter_:
        condition = filter_["condition"]
        rules = filter_.get("rules", [])
        if condition == "AND":
            mask = pd.Series(True, index=data.index)
            for rule in rules:
                mask = mask & _evaluate_filter(data, rule)
            return mask
        if condition == "OR":
            mask = pd.Series(False, index=data.index)
            for rule in rules:
                mask = mask | _evaluate_filter(data, rule)
            return mask
        raise BadUserInput(f"Condition: {condition} is not acceptable.")

    if "id" not in filter_:
        raise BadUserInput(
            "Invalid filters format. Filters did not contain the keys: 'condition' or 'id'."
        )

    column = filter_["id"]
    if column not in data.columns:
        raise BadUserInput(f"Column '{column}' does not exist in the input data.")

    return _evaluate_filter_rule(data[column], filter_)


def _evaluate_filter_rule(series: pd.Series, rule: Dict[str, Any]) -> pd.Series:
    operator = rule["operator"]
    value = rule.get("value")
    if operator == "equal":
        return series == value
    if operator == "not_equal":
        return series != value
    if operator == "less":
        return series < value
    if operator == "greater":
        return series > value
    if operator == "less_or_equal":
        return series <= value
    if operator == "greater_or_equal":
        return series >= value
    if operator == "between":
        return series.between(value[0], value[1])
    if operator == "not_between":
        return ~series.between(value[0], value[1])
    if operator == "is_null":
        return series.isnull()
    if operator == "is_not_null":
        return series.notnull()
    if operator == "in":
        return series.isin(value)
    if operator == "not_in":
        return ~series.isin(value)
    raise BadUserInput(f"Operator: {operator} is not acceptable.")
