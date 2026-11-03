import math
from typing import Dict

from exaflow import DType
from exaflow.worker_communication import CommonDataElement

_COMPARISON_SQL = {
    "equal": "=",
    "not_equal": "<>",
    "less": "<",
    "greater": ">",
    "less_or_equal": "<=",
    "greater_or_equal": ">=",
}
_RANGE_OPERATORS = {"between", "not_between"}
_MEMBERSHIP_OPERATORS = {"in", "not_in"}
_NULL_OPERATORS = {"is_null", "is_not_null"}

FILTER_OPERATORS = (
    set(_COMPARISON_SQL) | _RANGE_OPERATORS | _MEMBERSHIP_OPERATORS | _NULL_OPERATORS
)

__all__ = ["build_filter_clause", "validate_filter", "FilterError"]


class FilterError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def build_filter_clause(rules):
    """
    Convert a jQuery-format filter into a SQL clause and bound parameters.

    Identifiers are quoted. Values are returned as parameters and are never
    interpolated into the SQL text; the caller must execute the clause with
    those parameters. The value shape required by each operator is checked
    here. Column metadata is checked by validate_filter.
    """
    if rules is None:
        return None

    if not isinstance(rules, dict):
        raise FilterError("A dictionary should be provided to build a filter clause.")

    if "condition" in rules:
        _check_condition(rules["condition"])
        _check_group_rules(rules.get("rules"))
        cond = rules["condition"]
        parts = []
        params = []
        for rule in rules["rules"]:
            clause, clause_params = build_filter_clause(rule)
            parts.append(clause)
            params.extend(clause_params)
        return "(" + f" {cond} ".join(parts) + ")", params

    if "id" in rules:
        column_name = _quote_identifier(rules["id"])
        operator = rules["operator"]
        value = rules["value"]
        _check_operator_value(operator, value)
        return _operator_clause(column_name, operator, value)

    raise FilterError("Filters did not contain the keys: 'condition' or 'id'.")


def validate_filter(data_model: str, rules: dict, cdes: Dict[str, CommonDataElement]):
    """
    Validates a given filter in jQuery format.
    This function will check the validity of:
        1. The type of the filter
        2. The column name (if it exists in the metadata of the data_model)
        3. All the conditions that the filter contains
        4. All the operators that the filter contains
        5. The value shape required by each operator
        6. The type of the given value (if the column of the data_model and the value are the same type)
    """
    if rules is None:
        return

    _check_filter_type(rules)

    if "condition" in rules:
        _check_condition(rules["condition"])
        _check_group_rules(rules.get("rules"))
        rules = rules["rules"]
        for rule in rules:
            validate_filter(data_model, rule, cdes)
    elif "id" in rules:
        column_name = rules["id"]
        val = rules["value"]
        _check_operator(rules["operator"])
        _check_column_exists(data_model, column_name, cdes)
        _check_operator_value(rules["operator"], val)
        _check_value_type(column_name, val, cdes)
    else:
        raise FilterError(
            "Invalid filters format. Filters did not contain the keys: 'condition' or 'id'."
        )


def _quote_identifier(identifier) -> str:
    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'


def _operator_clause(column, operator, value):
    if operator in _COMPARISON_SQL:
        return f"{column} {_COMPARISON_SQL[operator]} ?", [value]
    if operator == "between":
        return f"{column} BETWEEN ? AND ?", list(value)
    if operator == "not_between":
        return f"NOT {column} BETWEEN ? AND ?", list(value)
    if operator == "is_null":
        return f"{column} IS NULL", []
    if operator == "is_not_null":
        return f"{column} IS NOT NULL", []
    placeholders = ", ".join("?" for _ in value)
    if operator == "in":
        return f"{column} IN ({placeholders})", list(value)
    if operator == "not_in":
        return f"{column} NOT IN ({placeholders})", list(value)
    raise FilterError(f"Operator: {operator} is not acceptable.")


def _check_filter_type(rules):
    if not isinstance(rules, dict):
        raise FilterError(f"Filter type can only be dict but was:{type(rules)}")


def _check_condition(condition: str):
    if condition not in ["OR", "AND"]:
        raise FilterError(f"Condition: {condition} is not acceptable.")


def _check_group_rules(rules):
    # An empty group would become "()" in the SQL text.
    if not isinstance(rules, list) or not rules:
        raise FilterError("A filter group requires a non-empty list of rules.")


def _check_operator(operator: str):
    if operator not in FILTER_OPERATORS:
        raise FilterError(f"Operator: {operator} is not acceptable.")


def _check_operator_value(operator, value):
    _check_operator(operator)
    if operator in _NULL_OPERATORS:
        if value is not None:
            raise FilterError(f"Operator {operator} only accepts None as its value.")
        return
    if value is None:
        raise FilterError("None is only allowed for is_null and is_not_null.")
    if operator in _COMPARISON_SQL:
        _check_scalar_value(value)
        return
    if operator in _RANGE_OPERATORS:
        if not isinstance(value, list) or len(value) != 2:
            raise FilterError(
                f"Operator {operator} requires a list of exactly two values."
            )
        for item in value:
            _check_scalar_value(item)
        return
    if operator in _MEMBERSHIP_OPERATORS:
        if not isinstance(value, list) or len(value) == 0:
            raise FilterError(
                f"Operator {operator} requires a non-empty list of values."
            )
        for item in value:
            _check_scalar_value(item)
        return
    raise FilterError(f"Operator: {operator} is not acceptable.")


def _check_scalar_value(value):
    if isinstance(value, list):
        raise FilterError(
            "Filter value must be a single int, float, or str, not a list."
        )
    if value is None:
        raise FilterError("None is only allowed for is_null and is_not_null.")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise FilterError(
            f"Filter value {value!r} should be of type int, float or str "
            f"but was {type(value)}."
        )
    if isinstance(value, str):
        if "\x00" in value:
            raise FilterError("Filter value must not contain a NUL byte.")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise FilterError(f"Filter value {value} must be a finite number.")


def _check_column_exists(data_model: str, column: str, cdes):
    if column not in cdes:
        raise FilterError(
            f"Column {column} does not exist in the metadata of the {data_model}!"
        )


def _check_value_type(column: str, value, cdes):
    if value is None:
        return

    if isinstance(value, list):
        [_check_value_type(column, item, cdes) for item in value]
    elif isinstance(value, (int, str, float)):
        _check_value_column_same_type(column, value, cdes)
    else:
        raise FilterError(
            f"Value {value} should be of type int, str, float but was {type(value)}"
        )


def _check_value_column_same_type(column, value, cdes):
    column_type = cdes[column].type
    dtype = DType.from_cde(column_type)
    if not _is_filter_value_compatible_with_dtype(value, dtype):
        raise FilterError(
            f"{column}'s type: {column_type} was different from the type of the given value:{type(value)}"
        )


def _is_filter_value_compatible_with_dtype(value, dtype: DType) -> bool:
    if dtype == DType.INT:
        return isinstance(value, int) and not isinstance(value, bool)

    if dtype == DType.FLOAT:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if dtype == DType.STR:
        return isinstance(value, str)

    return isinstance(value, dtype.to_py())
