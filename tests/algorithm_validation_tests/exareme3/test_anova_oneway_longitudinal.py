import numpy as np

from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "anova_oneway"


def make_test_input(visit1: str, visit2: str) -> dict:
    return {
        "inputdata": {
            "data_model": "longitudinal_dementia:0.1",
            "variables": ["agegroup", "lefthippocampus"],
            "datasets": [
                "longitudinal_dementia0",
                "longitudinal_dementia1",
                "longitudinal_dementia2",
            ],
            "filters": None,
        },
        "preprocessing": [
            {
                "name": "longitudinal_transformer",
                "parameters": {
                    "visit1": visit1,
                    "visit2": visit2,
                    "strategies": {
                        "lefthippocampus": "diff",
                        "agegroup": "first",
                    },
                },
            },
        ],
        "algorithm": {
            "x": ["agegroup"],
            "y": ["lefthippocampus"],
            "parameters": {},
        },
    }


def assert_valid_anova_response(result: dict):
    anova_table = result["anova_table"]
    tukey_test = result["tuckey_test"]

    assert anova_table["y_label"] == "lefthippocampus"
    assert anova_table["x_label"] == "agegroup"
    assert anova_table["n_obs"] > 0

    # one-way ANOVA identity: total df == n_obs - 1
    total_df = anova_table["df_residual"] + anova_table["df_explained"]
    assert np.isclose(total_df, anova_table["n_obs"] - 1)

    assert anova_table["f_stat"] >= 0
    assert 0 <= anova_table["p_value"] <= 1

    assert len(tukey_test) > 0
    for row in tukey_test:
        assert row["groupA"] != row["groupB"]
        assert 0 <= row["p_tuckey"] <= 1


def test_anova_oneway_longitudinal_bl_fl4():
    test_input = make_test_input(visit1="BL", visit2="FL4")
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    assert_valid_anova_response(result)


def test_anova_oneway_longitudinal_bl_fl1():
    test_input = make_test_input(visit1="BL", visit2="FL1")
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    assert_valid_anova_response(result)
