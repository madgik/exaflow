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
        """("test_age_value" = 17)""",
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
        """("test_pupil_reactivity_right_eye_result" <> 'Nonreactive')""",
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
        """("test_age_value" = 17 OR "test_pupil_reactivity_right_eye_result" <> 'Nonreactive')""",
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
        """("test_age_value" < 50 AND "test_age_value" > 20)""",
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
        """(NOT "test_age_value" BETWEEN 60 AND 90 AND "test_mortality_core" BETWEEN 0.3 AND 0.8)""",
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
        """("test_gose_score" IS NULL OR "test_gcs_total_score" IS NOT NULL)""",
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
        """("test_age_value" IN (17,19))""",
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
        """("test_gender_type" IN ('F','M'))""",
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
            """(("test_gender_type" = 'F' AND ("test_age_value" BETWEEN 20 AND 30 AND "test_gose_score" IS NOT NULL)) OR ("test_gender_type" <> 'F' AND ("test_mortality_core" > 0.5 AND "test_mortality_core" <= 0.8)))"""
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
