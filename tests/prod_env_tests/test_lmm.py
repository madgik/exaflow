import re

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import analysis_request
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

LMM_CASES = [
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
                        "righthippocampus",
                    ],
                },
                "test_case_num": 0,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala", "dataset"],
                    "y": ["righthippocampus"],
                    "parameters": {
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
            "name": "minimal_valid_single_fixed_plus_group",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": ["lefthippocampus", "dataset", "righthippocampus"],
                },
                "test_case_num": 1,
                "algorithm": {
                    "x": ["lefthippocampus", "dataset"],
                    "y": ["righthippocampus"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                    },
                },
            },
            "expected_status": 200,
        },
        id="minimal_valid_single_fixed_plus_group",
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
                        "righthippocampus",
                    ],
                },
                "test_case_num": 2,
                "algorithm": {
                    "x": ["lefthippocampus", "gender", "dataset"],
                    "y": ["righthippocampus"],
                    "parameters": {
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
                        "gender",
                        "agegroup",
                        "dataset",
                        "righthippocampus",
                    ],
                },
                "test_case_num": 3,
                "algorithm": {
                    "x": [
                        "lefthippocampus",
                        "gender",
                        "agegroup",
                        "dataset",
                    ],
                    "y": ["righthippocampus"],
                    "parameters": {
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
                        "righthippocampus",
                    ],
                },
                "test_case_num": 4,
                "algorithm": {
                    "x": ["lefthippocampus", "subjectageyears", "dataset"],
                    "y": ["righthippocampus"],
                    "parameters": {
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
                        "righthippocampus",
                    ],
                },
                "test_case_num": 5,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala", "gender"],
                    "y": ["righthippocampus"],
                    "parameters": {
                        "grouping_var": ["gender"],
                    },
                },
            },
            "expected_status": 200,
        },
        id="non_dataset_grouping_gender",
    ),
    pytest.param(
        {
            "name": "non_dataset_grouping_agegroup",
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
                        "leftamygdala",
                        "agegroup",
                        "righthippocampus",
                    ],
                },
                "test_case_num": 6,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala", "agegroup"],
                    "y": ["righthippocampus"],
                    "parameters": {
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
            "name": "grouping_var_missing_from_x",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "leftamygdala",
                        "righthippocampus",
                    ],
                },
                "test_case_num": 7,
                "algorithm": {
                    "x": ["lefthippocampus", "leftamygdala"],
                    "y": ["righthippocampus"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                    },
                },
            },
            "expected_status": 460,
            "expected_message": (
                r"Parameter 'grouping_var'.*must match exactly one variable included in 'x'"
                r"|Grouping variable.*inputdata \['x'\].*should be one of the following"
                r"|Parameter 'grouping_var' must match variables included in "
                r"inputdata 'x'"
            ),
        },
        id="grouping_var_missing_from_x",
    ),
    pytest.param(
        {
            "name": "composite_dataset_gender_grouping",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                    "variables": [
                        "lefthippocampus",
                        "dataset",
                        "gender",
                        "righthippocampus",
                    ],
                },
                "test_case_num": 8,
                "algorithm": {
                    "x": ["lefthippocampus", "dataset", "gender"],
                    "y": ["righthippocampus"],
                    "parameters": {
                        "grouping_var": ["dataset", "gender"],
                    },
                },
            },
            "expected_status": 200,
        },
        id="composite_dataset_gender_grouping",
    ),
    pytest.param(
        {
            "name": "only_grouping_var_no_fixed_left",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                    "variables": ["dataset", "righthippocampus"],
                },
                "test_case_num": 9,
                "algorithm": {
                    "x": ["dataset"],
                    "y": ["righthippocampus"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                    },
                },
            },
            "expected_status": 460,
            "expected_message": r"Algorithm input 'Covariates and grouping variable' should include at least 2 values.",
        },
        id="only_grouping_var_no_fixed_left",
    ),
    pytest.param(
        {
            "name": "invalid_categorical_y",
            "request": {
                "inputdata": {
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                    "variables": ["lefthippocampus", "dataset", "gender"],
                },
                "test_case_num": 10,
                "algorithm": {
                    "x": ["lefthippocampus", "dataset"],
                    "y": ["gender"],
                    "parameters": {
                        "grouping_var": ["dataset"],
                    },
                },
            },
            "expected_status": 460,
            "expected_message": (
                r"The CDE 'gender'.*should NOT be categorical"
                r"|The CDE 'gender'.*doesn't have one of the allowed types"
            ),
        },
        id="invalid_categorical_y",
    ),
]


@pytest.mark.parametrize("case", LMM_CASES)
def test_lmm_wrapper(case):
    response = analysis_request("lmm", case["request"])
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
        assert len(result["std_err"]) == len(result["indep_vars"])
        assert len(result["t_stats"]) == len(result["indep_vars"])
        assert result["pvalue_label"] == "P(>|t|)"
        assert len(result["pvalues"]) == len(result["indep_vars"])
        assert len(result["pvalues_display"]) == len(result["indep_vars"])
        assert len(result["lower_ci"]) == len(result["indep_vars"])
        assert len(result["upper_ci"]) == len(result["indep_vars"])

        assert result["n_obs"] > 0
        assert result["n_groups"] > 0
        assert result["df_model"] >= 0
        assert result["df_resid"] >= 0
        assert result["n_iter"] >= 1

        assert "sigma2" in result
        assert "sigma_u2" in result
        assert "ll_reml" in result
        assert "aic" in result
        assert "bic" in result
        assert "converged" in result

        if case["name"] in {
            "multiple_categorical_fixed_effects",
        }:
            assert len(result["indep_vars"]) > 3
        if case["name"] in {
            "non_dataset_grouping_gender",
            "non_dataset_grouping_agegroup",
        }:
            assert result["grouping_var"] != ["dataset"]
        if case["name"] == "composite_dataset_gender_grouping":
            assert "gender" not in result["indep_vars"]
            assert result["grouping_var"] == ["dataset", "gender"]
    else:
        assert re.search(case["expected_message"], response.text), response.text
