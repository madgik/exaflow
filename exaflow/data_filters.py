from typing import Dict

from exaflow import DType
from exaflow.worker_communication import CommonDataElement

FILTER_OPERATORS = {
    "equal": lambda column, value: f"{column} = {value}",
    "not_equal": lambda column, value: f"{column} <> {value}",
    "less": lambda column, value: f"{column} < {value}",
    "greater": lambda column, value: f"{column} > {value}",
    "less_or_equal": lambda column, value: f"{column} <= {value}",
    "greater_or_equal": lambda column, value: f"{column} >= {value}",
    "between": lambda column, values: f"{column} BETWEEN {values[0]} AND {values[1]}",
    "not_between": lambda column,
    value: f"NOT {column} BETWEEN {value[0]} AND {value[1]}",
    "is_null": lambda column, value: f"{column} IS NULL",
    "is_not_null": lambda column, value: f"{column} IS NOT NULL",
    "in": lambda column,
    values: f"{column} IN ({','.join(str(value) for value in values)})",
    "not_in": lambda column,
    values: f"{column} NOT IN ({','.join(str(value) for value in values)})",
}

__all__ = ["build_filter_clause", "validate_filter", "FilterError"]


class FilterError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def build_filter_clause(rules):
    """
    Converts and returns a given filter in jQuery format to a sql clause.
    This function will not check the validity of the
    filters (the only exception is the SQL Injection which will be handled by pydantic)
    """
    if rules is None:
        return

    if not isinstance(rules, dict):
        raise FilterError("A dictionary should be provided to build a filter clause.")

    if "condition" in rules:
        _check_condition(rules["condition"])
        cond = rules["condition"]
        rules = rules["rules"]
        return "(" + f" {cond} ".join(build_filter_clause(rule) for rule in rules) + ")"

    if "id" in rules:
        column_name = f'"{rules["id"]}"'
        op = FILTER_OPERATORS[rules["operator"]]
        value = _format_value_if_string(rules["type"], rules["value"])
        return op(column_name, value)

    raise FilterError("Filters did not contain the keys: 'condition' or 'id'.")


def validate_filter(data_model: str, rules: dict, cdes: Dict[str, CommonDataElement]):
    """
    Validates a given filter in jQuery format.
    This function will check the validity of:
        1. The type of the filter
        2. The column name (if it exists in the metadata of the data_model)
        3. All the conditions that the filter contains
        4. All the operators that the filter contains
        5. The type of the given value (if the column of the data_model and the value are the same type)
    """
    if rules is None:
        return

    _check_filter_type(rules)

    if "condition" in rules:
        _check_condition(rules["condition"])
        rules = rules["rules"]
        for rule in rules:
            validate_filter(data_model, rule, cdes)
    elif "id" in rules:
        column_name = rules["id"]
        val = rules["value"]
        _check_operator(rules["operator"])
        _check_column_exists(data_model, column_name, cdes)
        _check_value_type(column_name, val, cdes)
    else:
        raise FilterError(
            "Invalid filters format. Filters did not contain the keys: 'condition' or 'id'."
        )


def _format_value_if_string(column_type, val):
    if column_type != "string":
        return val

    def _quote(value):
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    if isinstance(val, list):
        return [_quote(item) for item in val]
    return _quote(val)


def _check_filter_type(rules):
    if not isinstance(rules, dict):
        raise FilterError(f"Filter type can only be dict but was:{type(rules)}")


def _check_condition(condition: str):
    if condition not in ["OR", "AND"]:
        raise FilterError(f"Condition: {condition} is not acceptable.")


def _check_operator(operator: str):
    if operator not in FILTER_OPERATORS:
        raise FilterError(f"Operator: {operator} is not acceptable.")


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
    column_sql_type = cdes[column].sql_type
    dtype = DType.from_cde(column_sql_type)
    try:
        value = dtype.to_py()(value)
    except ValueError:
        raise FilterError(
            f"{column}'s type: {column_sql_type} was different from the type of the given value:{type(value)}"
        )
