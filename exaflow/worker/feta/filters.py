"""FETA-owned SQL filter construction; requests are validated before use."""

FILTER_OPERATORS = {
    "equal": lambda column, value: f"{column} = {value}",
    "not_equal": lambda column, value: f"{column} <> {value}",
    "less": lambda column, value: f"{column} < {value}",
    "greater": lambda column, value: f"{column} > {value}",
    "less_or_equal": lambda column, value: f"{column} <= {value}",
    "greater_or_equal": lambda column, value: f"{column} >= {value}",
    "between": lambda column, values: f"{column} BETWEEN {values[0]} AND {values[1]}",
    "not_between": lambda column, value: (
        f"NOT {column} BETWEEN {value[0]} AND {value[1]}"
    ),
    "is_null": lambda column, value: f"{column} IS NULL",
    "is_not_null": lambda column, value: f"{column} IS NOT NULL",
    "in": lambda column, values: (
        f"{column} IN ({','.join(str(value) for value in values)})"
    ),
    "not_in": lambda column, values: (
        f"{column} NOT IN ({','.join(str(value) for value in values)})"
    ),
}


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


def _format_value_if_string(column_type, val):
    if column_type != "string":
        return val

    def _quote(value):
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    if isinstance(val, list):
        return [_quote(item) for item in val]
    return _quote(val)


def _check_condition(condition: str):
    if condition not in ["OR", "AND"]:
        raise FilterError(f"Condition: {condition} is not acceptable.")
