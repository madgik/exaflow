import re

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

LMM_CASES = [
    pytest.param(
        {
            "name": "basic_dataset_grouping",
            "request": {
                "inputdata": {
                    "y": ["righthippocampus"],
                    "x": ["lefthippocampus", "leftamygdala", "dataset"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "desd-synthdata0"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "dataset"},
                "test_case_num": 0,
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
                    "y": ["righthippocampus"],
                    "x": ["lefthippocampus", "dataset"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "dataset"},
                "test_case_num": 1,
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
                    "y": ["righthippocampus"],
                    "x": ["lefthippocampus", "gender", "dataset"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "dataset"},
                "test_case_num": 2,
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
                    "y": ["righthippocampus"],
                    "x": ["lefthippocampus", "gender", "agegroup", "dataset"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2", "desd-synthdata1"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "dataset"},
                "test_case_num": 3,
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
                    "y": ["righthippocampus"],
                    "x": ["lefthippocampus", "subjectageyears", "dataset"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "desd-synthdata1"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "dataset"},
                "test_case_num": 4,
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
                    "y": ["righthippocampus"],
                    "x": ["lefthippocampus", "leftamygdala", "gender"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "gender"},
                "test_case_num": 5,
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
                    "y": ["righthippocampus"],
                    "x": ["lefthippocampus", "leftamygdala", "agegroup"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2", "desd-synthdata1"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "agegroup"},
                "test_case_num": 6,
            },
            "expected_status": 200,
        },
        id="non_dataset_grouping_agegroup",
    ),
    pytest.param(
        {
            "name": "composite_dataset_gender_grouping",
            "request": {
                "inputdata": {
                    "y": ["righthippocampus"],
                    "x": ["lefthippocampus", "dataset", "gender"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1", "ppmi2"],
                    "filters": None,
                },
                "parameters": {"grouping_var": ["dataset", "gender"]},
                "test_case_num": 7,
            },
            "expected_status": 200,
        },
        id="composite_dataset_gender_grouping",
    ),
    pytest.param(
        {
            "name": "grouping_var_missing_from_x",
            "request": {
                "inputdata": {
                    "y": ["righthippocampus"],
                    "x": ["lefthippocampus", "leftamygdala"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "dataset"},
                "test_case_num": 8,
            },
            "expected_status": 460,
            "expected_message": (
                r"Parameter 'grouping_var'.*must match exactly one variable "
                r"included in 'x'"
                r"|Grouping variable.*inputdata \['x'\].*should be one of the following"
                r"|Parameter 'grouping_var' must match variables included in "
                r"inputdata 'x'"
            ),
        },
        id="grouping_var_missing_from_x",
    ),
    pytest.param(
        {
            "name": "only_grouping_var_no_fixed_left",
            "request": {
                "inputdata": {
                    "y": ["righthippocampus"],
                    "x": ["dataset"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "dataset"},
                "test_case_num": 9,
            },
            "expected_status": 460,
            "expected_message": (
                r"Inputdata 'Covariates and grouping variable' should include "
                r"at least 2 values."
                r"|Inputdata 'Covariates and grouping variable' must include at least "
                r"one fixed-effect covariate."
            ),
        },
        id="only_grouping_var_no_fixed_left",
    ),
    pytest.param(
        {
            "name": "invalid_categorical_y",
            "request": {
                "inputdata": {
                    "y": ["gender"],
                    "x": ["lefthippocampus", "dataset"],
                    "data_model": "dementia:0.1",
                    "datasets": ["ppmi0", "ppmi1"],
                    "filters": None,
                },
                "parameters": {"grouping_var": "dataset"},
                "test_case_num": 10,
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
    response = algorithm_request("lmm", case["request"])
    assert response.status_code == case["expected_status"], response.text

    if case["expected_status"] == 200:
        result = parse_response(response)

        assert result["dependent_var"] == case["request"]["inputdata"]["y"][0]
        grouping_var = case["request"]["parameters"]["grouping_var"]
        expected_grouping_var = (
            grouping_var[0] if len(grouping_var) == 1 else grouping_var
        )
        assert result["grouping_var"] == expected_grouping_var
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
            assert result["grouping_var"] != "dataset"
        if case["name"] == "composite_dataset_gender_grouping":
            assert "gender" not in result["indep_vars"]
            assert result["grouping_var"] == ["dataset", "gender"]
    else:
        assert re.search(case["expected_message"], response.text), response.text
