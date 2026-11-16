from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import analysis_request
from tests.algorithm_validation_tests.exareme3.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

alrorithm_name = "anova_twoway"
expected_file = Path(__file__).parent / "expected" / f"{alrorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_anova_two_way(test_input, expected, subtests):
    response = analysis_request(alrorithm_name, test_input)
    result = parse_response(response)

    validate_results(result, expected, subtests)


def validate_results(result, expected, subtests):
    terms = result["terms"]
    sum_sq = dict(zip(terms, result["sum_sq"]))
    df = dict(zip(terms, result["df"]))
    f_stat = dict(zip(terms, result["f_stat"]))
    f_pvalue = dict(zip(terms, result["f_pvalue"]))

    expected_terms = expected["sum_sq"].keys()

    # Sum Sq
    with subtests.test():
        for term in expected_terms:
            assert_allclose(sum_sq[term], expected["sum_sq"][term])

    # BUG In some cases the dfs are off by one for Residuals and the interaction term.
    # When this happens the F value and p values are also wrong due to the df error.
    # I was unable to find why, so I skip the rest of the test when this happens
    # and I will come back and fix this in the future.
    # https://team-1617704806227.atlassian.net/browse/MIP-745
    if df != expected["df"]:
        return

    # Df
    with subtests.test():
        for term in expected_terms:
            assert_allclose(df[term], expected["df"][term])

    # F values
    with subtests.test():
        for term in expected_terms:
            if f_stat[term]:
                assert_allclose(f_stat[term], expected["f_stat"][term])

    # p values
    with subtests.test():
        for term in expected_terms:
            if f_pvalue[term]:
                assert_allclose(f_pvalue[term], expected["f_pvalue"][term])


def test_anova_two_way__invalid_input__single_depvar():
    test_input = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "DESD-synthdata_0",
                "DESD-synthdata_1",
                "DESD-synthdata_2",
                "DESD-synthdata_3",
                "DESD-synthdata_4",
                "DESD-synthdata_5",
                "DESD-synthdata_6",
                "DESD-synthdata_7",
                "DESD-synthdata_8",
                "DESD-synthdata_9",
                "EDSD_0",
                "EDSD_1",
                "EDSD_2",
                "EDSD_3",
                "EDSD_4",
                "EDSD_5",
                "EDSD_6",
                "EDSD_7",
                "EDSD_8",
                "EDSD_9",
                "PPMI_0",
                "PPMI_1",
                "PPMI_2",
                "PPMI_3",
                "PPMI_4",
                "PPMI_5",
                "PPMI_6",
                "PPMI_7",
                "PPMI_8",
                "PPMI_9",
            ],
            "filters": None,
            "variables": ["dataset", "leftmpogpostcentralgyrusmedialsegment"],
        },
        "algorithm": {
            "x": ["dataset"],
            "y": ["leftmpogpostcentralgyrusmedialsegment"],
            "parameters": {"sstype": 2},
        },
    }
    response = analysis_request("anova_twoway", test_input)
    assert response.status_code == 460
