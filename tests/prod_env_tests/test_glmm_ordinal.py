import re

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

GLMM_ORDINAL_CASES = [
    pytest.param(
        {
            "name": "basic_5_level_dataset_grouping",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "leftamygdala",
                        "dataset",
                        "agegroup",
                    ],
                },
                "test_case_num": 0,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala", "dataset"],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                        "category_order": [
                            "-50y",
                            "50-59y",
                            "60-69y",
                            "70-79y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 200,
        },
        id="basic_5_level_dataset_grouping",
    ),
    pytest.param(
        {
            "name": "minimal_valid",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": ["lefthippocampus", "dataset", "agegroup"],
                },
                "test_case_num": 1,
                "algorithm": {
                    "x": ["lefthippocampus", "dataset"],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                        "category_order": [
                            "-50y",
                            "50-59y",
                            "60-69y",
                            "70-79y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 200,
        },
        id="minimal_valid",
    ),
    pytest.param(
        {
            "name": "categorical_fixed_effect_one_hot",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "gender",
                        "dataset",
                        "agegroup",
                    ],
                },
                "test_case_num": 2,
                "algorithm": {
                    "x": ["lefthippocampus", "gender", "dataset"],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                        "category_order": [
                            "-50y",
                            "50-59y",
                            "60-69y",
                            "70-79y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 200,
        },
        id="categorical_fixed_effect_one_hot",
    ),
    pytest.param(
        {
            "name": "multiple_categorical_fixed_effects",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": [
                        "ppmi0",
                        "ppmi1",
                        "ppmi2",
                        "desd-synthdata1",
                    ],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "gender",
                        "alzheimerbroadcategory",
                        "dataset",
                        "agegroup",
                    ],
                },
                "test_case_num": 3,
                "algorithm": {
                    "x": [
                        "lefthippocampus",
                        "gender",
                        "alzheimerbroadcategory",
                        "dataset",
                    ],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                        "category_order": [
                            "-50y",
                            "50-59y",
                            "60-69y",
                            "70-79y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 200,
        },
        id="multiple_categorical_fixed_effects",
    ),
    pytest.param(
        {
            "name": "numeric_age_covariate",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "desd-synthdata1"],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "subjectageyears",
                        "dataset",
                        "agegroup",
                    ],
                },
                "test_case_num": 4,
                "algorithm": {
                    "x": ["lefthippocampus", "subjectageyears", "dataset"],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                        "category_order": [
                            "-50y",
                            "50-59y",
                            "60-69y",
                            "70-79y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 200,
        },
        id="numeric_age_covariate",
    ),
    pytest.param(
        {
            "name": "non_dataset_grouping_gender",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "leftamygdala",
                        "gender",
                        "agegroup",
                    ],
                },
                "test_case_num": 5,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala", "gender"],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["gender"],
                        "category_order": [
                            "-50y",
                            "50-59y",
                            "60-69y",
                            "70-79y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 200,
        },
        id="non_dataset_grouping_gender",
    ),
    pytest.param(
        {
            "name": "non_dataset_grouping_alzheimer_category",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "desd-synthdata1"],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "leftamygdala",
                        "alzheimerbroadcategory",
                        "agegroup",
                    ],
                },
                "test_case_num": 6,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala", "alzheimerbroadcategory"],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["alzheimerbroadcategory"],
                        "category_order": [
                            "-50y",
                            "50-59y",
                            "60-69y",
                            "70-79y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 200,
        },
        id="non_dataset_grouping_alzheimer_category",
    ),
    pytest.param(
        {
            "name": "incomplete_category_order",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": ["lefthippocampus", "dataset", "agegroup"],
                },
                "test_case_num": 7,
                "algorithm": {
                    "x": ["lefthippocampus", "dataset"],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                        "category_order": [
                            "50-59y",
                            "60-69y",
                            "70-79y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 460,
            "expected_message": r"category_order.*does not cover all observed y categories",
        },
        id="incomplete_category_order",
    ),
    pytest.param(
        {
            "name": "duplicate_category_order",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": ["lefthippocampus", "dataset", "agegroup"],
                },
                "test_case_num": 8,
                "algorithm": {
                    "x": ["lefthippocampus", "dataset"],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                        "category_order": [
                            "-50y",
                            "50-59y",
                            "60-69y",
                            "60-69y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 460,
            "expected_message": r"Parameter 'category_order' must not contain duplicates",
        },
        id="duplicate_category_order",
    ),
    pytest.param(
        {
            "name": "grouping_var_missing_from_x",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                    "variables": ["lefthippocampus", "leftamygdala", "agegroup"],
                },
                "test_case_num": 9,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala"],
                    "y": ["agegroup"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                        "category_order": [
                            "-50y",
                            "50-59y",
                            "60-69y",
                            "70-79y",
                            "+80y",
                        ],
                    },
                },
            },
            "expected_status": 460,
            "expected_message": (
                r"Parameter 'grouping_var'.*must match exactly one variable included in 'x'"
                r"|Grouping variable.*inputdata \['x'\].*should be one of the following"
            ),
        },
        id="grouping_var_missing_from_x",
    ),
]


@pytest.mark.parametrize("case", GLMM_ORDINAL_CASES)
def test_glmm_ordinal_wrapper(case):
    response = algorithm_request("glmm_ordinal", case["request"])
    assert response.status_code == case["expected_status"], response.text

    if case["expected_status"] == 200:
        result = parse_response(response)

        assert result["dependent_var"] == case["request"]["algorithm"]["y"][0]
        assert (
            result["grouping_var"]
            == case["request"]["algorithm"]["parameters"]["grouping_var"]
        )
        assert result["category_order"] == [
            str(x) for x in case["request"]["algorithm"]["parameters"]["category_order"]
        ]
        assert result["indep_vars"][0] == "Intercept"
        assert len(result["coefficients"]) == len(result["indep_vars"])
        assert len(result["cutpoints"]) == len(result["category_order"]) - 1
        assert result["n_obs"] > 0
        assert result["n_groups"] > 0
        assert result["n_iter"] >= 1

        assert "sigma_u2" in result
        assert "converged" in result

        if case["name"] == "multiple_categorical_fixed_effects":
            assert len(result["indep_vars"]) > 3
    else:
        assert re.search(case["expected_message"], response.text), response.text
