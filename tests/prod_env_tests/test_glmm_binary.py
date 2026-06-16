import re

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

GLMM_BINARY_CASES = [
    pytest.param(
        {
            "name": "basic_dataset_grouping",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "desd-synthdata0"],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "leftamygdala",
                        "dataset",
                        "gender",
                    ],
                },
                "test_case_num": 0,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala", "dataset"],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "F",
                        "grouping_var": ["dataset"],
                    },
                },
            },
            "expected_status": 200,
        },
        id="basic_dataset_grouping",
    ),
    pytest.param(
        {
            "name": "minimal_valid",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": ["lefthippocampus", "dataset", "gender"],
                },
                "test_case_num": 1,
                "algorithm": {
                    "x": ["lefthippocampus", "dataset"],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "F",
                        "grouping_var": ["dataset"],
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
                        "agegroup",
                        "dataset",
                        "gender",
                    ],
                },
                "test_case_num": 2,
                "algorithm": {
                    "x": ["lefthippocampus", "agegroup", "dataset"],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "F",
                        "grouping_var": ["dataset"],
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
                        "agegroup",
                        "alzheimerbroadcategory",
                        "dataset",
                        "gender",
                    ],
                },
                "test_case_num": 3,
                "algorithm": {
                    "x": [
                        "lefthippocampus",
                        "agegroup",
                        "alzheimerbroadcategory",
                        "dataset",
                    ],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "F",
                        "grouping_var": ["dataset"],
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
                        "gender",
                    ],
                },
                "test_case_num": 4,
                "algorithm": {
                    "x": ["lefthippocampus", "subjectageyears", "dataset"],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "F",
                        "grouping_var": ["dataset"],
                    },
                },
            },
            "expected_status": 200,
        },
        id="numeric_age_covariate",
    ),
    pytest.param(
        {
            "name": "non_dataset_grouping_agegroup",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "leftamygdala",
                        "agegroup",
                        "gender",
                    ],
                },
                "test_case_num": 5,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala", "agegroup"],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "F",
                        "grouping_var": ["agegroup"],
                    },
                },
            },
            "expected_status": 200,
        },
        id="non_dataset_grouping_agegroup",
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
                        "gender",
                    ],
                },
                "test_case_num": 6,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala", "alzheimerbroadcategory"],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "F",
                        "grouping_var": ["alzheimerbroadcategory"],
                    },
                },
            },
            "expected_status": 200,
        },
        id="non_dataset_grouping_alzheimer_category",
    ),
    pytest.param(
        {
            "name": "invalid_positive_class",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                    "variables": ["lefthippocampus", "dataset", "gender"],
                },
                "test_case_num": 7,
                "algorithm": {
                    "x": ["lefthippocampus", "dataset"],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "X",
                        "grouping_var": ["dataset"],
                    },
                },
            },
            "expected_status": 460,
            "expected_message": r"Positive class.*should be one of the following",
        },
        id="invalid_positive_class",
    ),
    pytest.param(
        {
            "name": "grouping_var_missing_from_x",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                    "variables": ["lefthippocampus", "leftamygdala", "gender"],
                },
                "test_case_num": 8,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala"],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "F",
                        "grouping_var": ["dataset"],
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
    pytest.param(
        {
            "name": "only_grouping_var_no_fixed_left",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                    "variables": ["dataset", "gender"],
                },
                "test_case_num": 9,
                "algorithm": {
                    "x": ["dataset"],
                    "y": ["gender"],
                    "parameters": {
                        "positive_class": "F",
                        "grouping_var": ["dataset"],
                    },
                },
            },
            "expected_status": 460,
            "expected_message": r"Inputdata 'Covariates and grouping variable' should include at least 2 values.",
        },
        id="only_grouping_var_no_fixed_left",
    ),
]


@pytest.mark.parametrize("case", GLMM_BINARY_CASES)
def test_glmm_binary_wrapper(case):
    response = algorithm_request("glmm_binary", case["request"])
    assert response.status_code == case["expected_status"], response.text

    if case["expected_status"] == 200:
        result = parse_response(response)

        assert result["dependent_var"] == case["request"]["algorithm"]["y"][0]
        assert (
            result["grouping_var"]
            == case["request"]["algorithm"]["parameters"]["grouping_var"]
        )
        assert result["indep_vars"][0] == "Intercept"
        assert len(result["coefficients"]) == len(result["indep_vars"])
        assert result["n_obs"] > 0
        assert result["n_groups"] > 0
        assert result["n_iter"] >= 1

        assert "sigma_u2" in result
        assert "converged" in result

        if case["name"] in {
            "categorical_fixed_effect_one_hot",
            "multiple_categorical_fixed_effects",
        }:
            assert len(result["indep_vars"]) > 3
    else:
        assert re.search(case["expected_message"], response.text), response.text
