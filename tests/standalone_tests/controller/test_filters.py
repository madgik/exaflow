import pytest

from exaflow.data_filters import FilterError
from exaflow.data_filters import build_filter_clause
from exaflow.data_filters import validate_filter
from exaflow.worker_communication import CommonDataElement

DATA_MODEL = "test_data_model1:0.1"


@pytest.fixture
def data_models():
    return {
        "test_data_model1:0.1": {
            "test_age_value": CommonDataElement(
                code="test_age_value",
                label="test test_age_value",
                sql_type="int",
                is_categorical=False,
                enumerations=None,
                min=0,
                max=130,
            ),
            "test_pupil_reactivity_right_eye_result": CommonDataElement(
                code="test_pupil_reactivity_right_eye_result",
                label="test test_pupil_reactivity_right_eye_result",
                sql_type="text",
                is_categorical=True,
                enumerations={
                    "Sluggish": "Sluggish",
                    "Nonreactive": "Nonreactive",
                    "Brisk": "Brisk",
                    "Untestable": "Untestable",
                    "Unknown": "Unknown",
                },
                min=None,
                max=None,
            ),
            "test_mortality_core": CommonDataElement(
                code="test_mortality_core",
                label="test_mortality_core",
                sql_type="real",
                is_categorical=False,
                enumerations=None,
                min=None,
                max=None,
            ),
            "test_gose_score": CommonDataElement(
                code="test_gose_score",
                label="test_gose_score",
                sql_type="text",
                is_categorical=True,
                enumerations={
                    "1": "Dead",
                    "2": "Vegetative State",
                    "3": "Lower Severe Disability",
                    "4": "Upper Severe Disability",
                    "5": "Lower Moderate Disability",
                    "6": "Upper Moderate Disability",
                    "7": "Lower Good Recovery",
                    "8": "Upper Good Recovery",
                },
                min=None,
                max=None,
            ),
            "test_gcs_total_score": CommonDataElement(
                code="test_gcs_total_score",
                label="test_gcs_total_score",
                sql_type="text",
                is_categorical=True,
                enumerations={
                    "3": "3",
                    "4": "4",
                    "5": "5",
                    "6": "6",
                    "7": "7",
                    "8": "8",
                    "9": "9",
                    "10": "10",
                    "11": "11",
                    "12": "12",
                    "13": "13",
                    "14": "14",
                    "15": "15",
                    "untestable": "untestable",
                    "unknown": "unknown",
                },
                min=None,
                max=None,
            ),
            "test_gender_type": CommonDataElement(
                code="test_gender_type",
                label="test_gender_type",
                sql_type="text",
                is_categorical=True,
                enumerations={
                    "M": "Male",
                    "F": "Female",
                },
                min=None,
                max=None,
            ),
        },
    }


all_success_cases = [
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "equal",
                    "value": 17,
                }
            ],
            "valid": True,
        },
        ('("test_age_value" = ?)', [17]),
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_pupil_reactivity_right_eye_result",
                    "field": "test_pupil_reactivity_right_eye_result",
                    "type": "string",
                    "input": "text",
                    "operator": "not_equal",
                    "value": "Nonreactive",
                }
            ],
            "valid": True,
        },
        (
            '("test_pupil_reactivity_right_eye_result" <> ?)',
            ["Nonreactive"],
        ),
    ),
    (
        {
            "condition": "OR",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "equal",
                    "value": 17,
                },
                {
                    "id": "test_pupil_reactivity_right_eye_result",
                    "field": "test_pupil_reactivity_right_eye_result",
                    "type": "string",
                    "input": "text",
                    "operator": "not_equal",
                    "value": "Nonreactive",
                },
            ],
            "valid": True,
        },
        (
            '("test_age_value" = ? OR "test_pupil_reactivity_right_eye_result" <> ?)',
            [17, "Nonreactive"],
        ),
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "less",
                    "value": 50,
                },
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "greater",
                    "value": 20,
                },
            ],
            "valid": True,
        },
        ('("test_age_value" < ? AND "test_age_value" > ?)', [50, 20]),
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "not_between",
                    "value": [60, 90],
                },
                {
                    "id": "test_mortality_core",
                    "field": "test_mortality_core",
                    "type": "double",
                    "input": "number",
                    "operator": "between",
                    "value": [0.3, 0.8],
                },
            ],
            "valid": True,
        },
        (
            '(NOT "test_age_value" BETWEEN ? AND ? AND '
            '"test_mortality_core" BETWEEN ? AND ?)',
            [60, 90, 0.3, 0.8],
        ),
    ),
    (
        {
            "condition": "OR",
            "rules": [
                {
                    "id": "test_gose_score",
                    "field": "test_gose_score",
                    "type": "text",
                    "input": "text",
                    "operator": "is_null",
                    "value": None,
                },
                {
                    "id": "test_gcs_total_score",
                    "field": "test_gcs_total_score",
                    "type": "int",
                    "input": "number",
                    "operator": "is_not_null",
                    "value": None,
                },
            ],
            "valid": True,
        },
        (
            '("test_gose_score" IS NULL OR "test_gcs_total_score" IS NOT NULL)',
            [],
        ),
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_age_value",
                    "field": "test_age_value",
                    "type": "integer",
                    "input": "number",
                    "operator": "in",
                    "value": [17, 19],
                }
            ],
            "valid": True,
        },
        ('("test_age_value" IN (?, ?))', [17, 19]),
    ),
    (
        {
            "condition": "AND",
            "rules": [
                {
                    "id": "test_gender_type",
                    "field": "test_gender_type",
                    "type": "string",
                    "input": "text",
                    "operator": "in",
                    "value": ["F", "M"],
                }
            ],
            "valid": True,
        },
        ('("test_gender_type" IN (?, ?))', ["F", "M"]),
    ),
    (
        {
            "condition": "OR",
            "rules": [
                {
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "test_gender_type",
                            "field": "test_gender_type",
                            "type": "string",
                            "input": "text",
                            "operator": "equal",
                            "value": "F",
                        },
                        {
                            "condition": "AND",
                            "rules": [
                                {
                                    "id": "test_age_value",
                                    "field": "test_age_value",
                                    "type": "int",
                                    "input": "number",
                                    "operator": "between",
                                    "value": [20, 30],
                                },
                                {
                                    "id": "test_gose_score",
                                    "field": "test_gose_score",
                                    "type": "text",
                                    "input": "text",
                                    "operator": "is_not_null",
                                    "value": None,
                                },
                            ],
                        },
                    ],
                },
                {
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "test_gender_type",
                            "field": "test_gender_type",
                            "type": "string",
                            "input": "text",
                            "operator": "not_equal",
                            "value": "F",
                        },
                        {
                            "condition": "AND",
                            "rules": [
                                {
                                    "id": "test_mortality_core",
                                    "field": "test_mortality_core",
                                    "type": "double",
                                    "input": "number",
                                    "operator": "greater",
                                    "value": 0.5,
                                },
                                {
                                    "id": "test_mortality_core",
                                    "field": "test_mortality_core",
                                    "type": "double",
                                    "input": "number",
                                    "operator": "less_or_equal",
                                    "value": 0.8,
                                },
                            ],
                        },
                    ],
                },
            ],
            "valid": True,
        },
        (
            "("
            '("test_gender_type" = ? AND ("test_age_value" BETWEEN ? AND ? AND '
            '"test_gose_score" IS NOT NULL)) OR ("test_gender_type" <> ? AND '
            '("test_mortality_core" > ? AND "test_mortality_core" <= ?))'
            ")",
            ["F", 20, 30, "F", 0.5, 0.8],
        ),
    ),
]


@pytest.mark.parametrize("test_input,expected", all_success_cases)
def test_build_filter_clause(test_input, expected):
    assert build_filter_clause(test_input) == expected


@pytest.mark.parametrize("test_input,expected", all_success_cases)
def test_validate_filter(test_input, expected, data_models):
    validate_filter(DATA_MODEL, test_input, data_models[DATA_MODEL])


@pytest.mark.parametrize(
    "column,value",
    [
        ("test_age_value", 17),
        ("test_mortality_core", 0.5),
        ("test_mortality_core", 1),
    ],
)
def test_validate_filter_accepts_numeric_json_values(column, value, data_models):
    validate_filter(
        DATA_MODEL,
        {
            "id": column,
            "field": column,
            "type": "int",
            "input": "number",
            "operator": "equal",
            "value": value,
        },
        data_models[DATA_MODEL],
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("test_age_value", "17"),
        ("test_mortality_core", "0.5"),
        ("test_age_value", True),
        ("test_mortality_core", True),
    ],
)
def test_validate_filter_rejects_non_numeric_json_values_for_numeric_cdes(
    column, value, data_models
):
    with pytest.raises(FilterError):
        validate_filter(
            DATA_MODEL,
            {
                "id": column,
                "field": column,
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": value,
            },
            data_models[DATA_MODEL],
        )


def test_validate_filter_rejects_non_numeric_json_values_in_numeric_lists(
    data_models,
):
    with pytest.raises(FilterError):
        validate_filter(
            DATA_MODEL,
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "in",
                "value": [17, "18"],
            },
            data_models[DATA_MODEL],
        )


all_build_filter_clause_fail_cases = [
    {
        "condition": "ANDOR",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": 17,
            }
        ],
        "valid": True,
    },
    0,
    "not_a_filter",
    {"data": 0},
    {0},
]


@pytest.mark.parametrize("test_input", all_build_filter_clause_fail_cases)
def test_build_filter_clause_fail_cases(test_input):
    with pytest.raises(FilterError):
        build_filter_clause(test_input)


all_validate_filter_fail_cases = [
    {
        "condition": "ANDOR",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": 17,
            }
        ],
        "valid": True,
    },
    0,
    "not_a_filter",
    {"data": 0},
    {0},
    {
        "condition": "AND",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "invalid operator",
                "value": 17,
            }
        ],
        "valid": True,
    },
    FilterError,
    {
        "condition": "AND",
        "rules": [
            {
                "id": "invalid column",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": 17,
            }
        ],
        "valid": True,
    },
    {
        "condition": "AND",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": "17",
            }
        ],
        "valid": True,
    },
    {
        "condition": "AND",
        "rules": [
            {
                "id": "test_age_value",
                "field": "test_age_value",
                "type": "int",
                "input": "number",
                "operator": "equal",
                "value": True,
            }
        ],
        "valid": True,
    },
]


@pytest.mark.parametrize("test_input", all_validate_filter_fail_cases)
def test_validate_filter_fail_cases_bad_filter(test_input, data_models):
    with pytest.raises(FilterError):
        validate_filter(DATA_MODEL, test_input, data_models)


def test_build_filter_clause_quotes_strings_regardless_of_declared_type():
    rule = {
        "id": "test_gender_type",
        "field": "test_gender_type",
        "type": "integer",
        "input": "text",
        "operator": "equal",
        "value": "'F' OR 1=1",
    }

    sql, params = build_filter_clause(rule)

    assert sql == '"test_gender_type" = ?'
    assert params == ["'F' OR 1=1"]


def test_build_filter_clause_escapes_column_identifier():
    rule = {
        "id": 'col" OR 1=1 --',
        "type": "string",
        "operator": "is_null",
        "value": None,
    }

    assert build_filter_clause(rule) == ('"col"" OR 1=1 --" IS NULL', [])


@pytest.mark.parametrize("value", [True, {"a": 1}, float("nan")])
def test_build_filter_clause_rejects_unsupported_values(value):
    rule = {
        "id": "test_age_value",
        "type": "integer",
        "operator": "equal",
        "value": value,
    }

    with pytest.raises(FilterError):
        build_filter_clause(rule)


def _expect_filter_error(callback):
    try:
        result = callback()
    except FilterError:
        return
    except Exception as exc:
        pytest.fail(f"expected FilterError, got {type(exc).__name__}: {exc}")
    pytest.fail(f"expected FilterError, got result {result!r}")


def _text_column_cdes():
    return {
        "s": CommonDataElement(
            code="s",
            label="s",
            sql_type="text",
            is_categorical=False,
            enumerations=None,
        )
    }


# Python list repr is interpolated into SQL and DuckDB does not treat \' as an
# escape, so this closes the literal and ORs a tautology.
INJECTION_PAYLOAD = {
    "id": "s",
    "operator": "equal",
    "value": ['][1]) OR (1=1) --"'],
}

SINGLE_VALUE_OPERATORS = [
    "equal",
    "not_equal",
    "less",
    "greater",
    "less_or_equal",
    "greater_or_equal",
]
NULL_OPERATORS = ["is_null", "is_not_null"]
RANGE_OPERATORS = ["between", "not_between"]
MEMBERSHIP_OPERATORS = ["in", "not_in"]


@pytest.mark.parametrize("operator", SINGLE_VALUE_OPERATORS)
def test_security_validate_filter_rejects_list_on_single_value_operator(
    operator, data_models
):
    _expect_filter_error(
        lambda: validate_filter(
            DATA_MODEL,
            {"id": "test_age_value", "operator": operator, "value": [1]},
            data_models[DATA_MODEL],
        )
    )


@pytest.mark.parametrize("operator", SINGLE_VALUE_OPERATORS)
def test_security_build_filter_clause_rejects_list_on_single_value_operator(operator):
    _expect_filter_error(
        lambda: build_filter_clause(
            {"id": "test_age_value", "operator": operator, "value": [1]}
        )
    )


def test_security_validate_filter_rejects_injection_payload():
    _expect_filter_error(
        lambda: validate_filter("dm", INJECTION_PAYLOAD, _text_column_cdes())
    )


def test_security_build_filter_clause_rejects_injection_payload():
    _expect_filter_error(lambda: build_filter_clause(INJECTION_PAYLOAD))


def test_security_validate_filter_rejects_nested_injection_payload():
    rules = {
        "condition": "OR",
        "rules": [
            {"id": "s", "operator": "equal", "value": "ok"},
            INJECTION_PAYLOAD,
        ],
    }
    _expect_filter_error(lambda: validate_filter("dm", rules, _text_column_cdes()))


def test_security_build_filter_clause_rejects_nested_injection_payload():
    rules = {
        "condition": "OR",
        "rules": [
            {"id": "s", "operator": "equal", "value": "ok"},
            INJECTION_PAYLOAD,
        ],
    }
    _expect_filter_error(lambda: build_filter_clause(rules))


@pytest.mark.parametrize(
    "group",
    [
        {"condition": "AND", "rules": []},
        {"condition": "AND"},
        {"condition": "OR", "rules": {"id": "s", "operator": "is_null"}},
        {"condition": "AND", "rules": [{"condition": "OR", "rules": []}]},
    ],
    ids=["empty", "missing", "not_a_list", "nested_empty"],
)
def test_security_filter_rejects_group_without_rules(group):
    _expect_filter_error(lambda: validate_filter("dm", group, _text_column_cdes()))
    _expect_filter_error(lambda: build_filter_clause(group))


@pytest.mark.parametrize("operator", RANGE_OPERATORS)
@pytest.mark.parametrize(
    "column,value",
    [
        ("test_gender_type", "abc"),
        ("test_age_value", [1]),
        ("test_age_value", 5),
        ("test_age_value", [1, 2, 3]),
    ],
)
def test_security_validate_filter_rejects_malformed_between(
    operator, column, value, data_models
):
    _expect_filter_error(
        lambda: validate_filter(
            DATA_MODEL,
            {"id": column, "operator": operator, "value": value},
            data_models[DATA_MODEL],
        )
    )


@pytest.mark.parametrize("operator", RANGE_OPERATORS)
@pytest.mark.parametrize("value", ["abc", [1], 5, [1, 2, 3]])
def test_security_build_filter_clause_rejects_malformed_between(operator, value):
    _expect_filter_error(
        lambda: build_filter_clause(
            {"id": "test_age_value", "operator": operator, "value": value}
        )
    )


@pytest.mark.parametrize("operator", MEMBERSHIP_OPERATORS)
@pytest.mark.parametrize(
    "column,value",
    [
        ("test_gender_type", "abc"),
        ("test_age_value", []),
        ("test_gender_type", []),
    ],
)
def test_security_validate_filter_rejects_malformed_membership(
    operator, column, value, data_models
):
    _expect_filter_error(
        lambda: validate_filter(
            DATA_MODEL,
            {"id": column, "operator": operator, "value": value},
            data_models[DATA_MODEL],
        )
    )


@pytest.mark.parametrize("operator", MEMBERSHIP_OPERATORS)
@pytest.mark.parametrize("value", ["abc", []])
def test_security_build_filter_clause_rejects_malformed_membership(operator, value):
    _expect_filter_error(
        lambda: build_filter_clause({"id": "s", "operator": operator, "value": value})
    )


@pytest.mark.parametrize(
    "operator",
    SINGLE_VALUE_OPERATORS + RANGE_OPERATORS + MEMBERSHIP_OPERATORS,
)
def test_security_validate_filter_rejects_none_value(operator, data_models):
    _expect_filter_error(
        lambda: validate_filter(
            DATA_MODEL,
            {"id": "test_age_value", "operator": operator, "value": None},
            data_models[DATA_MODEL],
        )
    )


@pytest.mark.parametrize(
    "operator,value",
    [
        ("in", [None]),
        ("not_in", [1, None]),
        ("between", [1, None]),
        ("not_between", [None, 2]),
    ],
)
def test_security_validate_filter_rejects_none_inside_list(
    operator, value, data_models
):
    _expect_filter_error(
        lambda: validate_filter(
            DATA_MODEL,
            {"id": "test_age_value", "operator": operator, "value": value},
            data_models[DATA_MODEL],
        )
    )


@pytest.mark.parametrize(
    "operator",
    SINGLE_VALUE_OPERATORS + RANGE_OPERATORS + MEMBERSHIP_OPERATORS,
)
def test_security_build_filter_clause_rejects_none_value(operator):
    _expect_filter_error(
        lambda: build_filter_clause(
            {"id": "test_age_value", "operator": operator, "value": None}
        )
    )


@pytest.mark.parametrize(
    "operator,value",
    [
        ("in", [None]),
        ("not_in", [1, None]),
        ("between", [1, None]),
        ("not_between", [None, 2]),
    ],
)
def test_security_build_filter_clause_rejects_none_inside_list(operator, value):
    _expect_filter_error(
        lambda: build_filter_clause(
            {"id": "test_age_value", "operator": operator, "value": value}
        )
    )


@pytest.mark.parametrize("operator", NULL_OPERATORS)
def test_security_null_operators_accept_none(operator, data_models):
    rule = {"id": "test_age_value", "operator": operator, "value": None}
    validate_filter(DATA_MODEL, rule, data_models[DATA_MODEL])
    result = build_filter_clause(rule)
    sql = result[0] if isinstance(result, tuple) else result
    expected = "IS NULL" if operator == "is_null" else "IS NOT NULL"
    assert expected in sql


@pytest.mark.parametrize(
    "operator,column,value",
    [
        ("is_null", "test_gender_type", "x"),
        ("is_not_null", "test_age_value", 0),
    ],
)
def test_security_null_operators_reject_non_none_values(
    operator, column, value, data_models
):
    _expect_filter_error(
        lambda: validate_filter(
            DATA_MODEL,
            {"id": column, "operator": operator, "value": value},
            data_models[DATA_MODEL],
        )
    )
    _expect_filter_error(
        lambda: build_filter_clause(
            {"id": column, "operator": operator, "value": value}
        )
    )


@pytest.mark.parametrize(
    "operator,value",
    [
        ("equal", "a\x00b"),
        ("in", ["ok", "a\x00b"]),
        ("between", ["a\x00", "b"]),
    ],
)
def test_security_validate_filter_rejects_nul_byte(operator, value):
    _expect_filter_error(
        lambda: validate_filter(
            "dm",
            {"id": "s", "operator": operator, "value": value},
            _text_column_cdes(),
        )
    )


@pytest.mark.parametrize(
    "operator,value",
    [
        ("equal", "a\x00b"),
        ("in", ["ok", "a\x00b"]),
        ("between", ["a\x00", "b"]),
    ],
)
def test_security_build_filter_clause_rejects_nul_byte(operator, value):
    _expect_filter_error(
        lambda: build_filter_clause({"id": "s", "operator": operator, "value": value})
    )


@pytest.mark.parametrize(
    "rule,expected_sql,expected_params",
    [
        (
            {"id": "s", "operator": "equal", "value": "O'Hara --"},
            '"s" = ?',
            ["O'Hara --"],
        ),
        (
            {"id": "age", "operator": "in", "value": [17, 19]},
            '"age" IN (?, ?)',
            [17, 19],
        ),
        (
            {"id": "age", "operator": "not_in", "value": ["a"]},
            '"age" NOT IN (?)',
            ["a"],
        ),
        (
            {"id": "age", "operator": "between", "value": [0.3, 0.8]},
            '"age" BETWEEN ? AND ?',
            [0.3, 0.8],
        ),
        (
            {"id": "age", "operator": "not_between", "value": [1, 2]},
            'NOT "age" BETWEEN ? AND ?',
            [1, 2],
        ),
        (
            {"id": "age", "operator": "less", "value": 5},
            '"age" < ?',
            [5],
        ),
    ],
)
def test_security_build_filter_clause_binds_parameters(
    rule, expected_sql, expected_params
):
    result = build_filter_clause(rule)
    assert isinstance(result, tuple), result
    sql, params = result
    assert sql == expected_sql
    assert params == expected_params


def test_security_build_filter_clause_binds_nested_parameters_in_order():
    rule = {
        "condition": "OR",
        "rules": [
            {"id": "s", "operator": "equal", "value": "a"},
            {
                "condition": "AND",
                "rules": [
                    {"id": "age", "operator": "between", "value": [1, 2]},
                    {"id": "s", "operator": "in", "value": ["x", "y"]},
                ],
            },
        ],
    }
    result = build_filter_clause(rule)
    assert isinstance(result, tuple), result
    sql, params = result
    assert sql == '("s" = ? OR ("age" BETWEEN ? AND ? AND "s" IN (?, ?)))'
    assert params == ["a", 1, 2, "x", "y"]
