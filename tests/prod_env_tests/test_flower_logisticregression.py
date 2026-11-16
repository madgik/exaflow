from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response

algorithm_name = "logistic_regression_fedaverage_flower"

import pytest


@pytest.mark.skip(reason="Flower tests disabled pending MIP-1203.")
def test_logisticregression_algorithm():
    test_input = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
                "PPMI_0",
                "PPMI_1",
                "PPMI_2",
                "PPMI_3",
                "PPMI_5",
                "PPMI_6",
                "EDSD_6",
                "PPMI_7",
                "PPMI_8",
                "PPMI_9",
            ],
            "validation_datasets": ["PPMI_TEST"],
            "filters": None,
            "variables": ["lefthippocampus", "gender"],
        },
        "algorithm": {
            "x": ["lefthippocampus"],
            "y": ["gender"],
            "parameters": {},
        },
    }
    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)

    # this test only ensures that the algorithm runs smoothly without errors
    assert result
