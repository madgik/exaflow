from tests.algorithm_validation_tests.exareme3.conftest import algorithm_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response

algorithm_name = "logistic_regression_fedaverage_flower"

import pytest


@pytest.mark.skip(reason="Flower tests disabled pending MIP-1203.")
def test_logisticregression_algorithm():
    test_input = {
        "inputdata": {
            "y": ["gender"],
            "x": ["lefthippocampus"],
            "data_model": "dementia:0.1",
            "datasets": [
                "ppmi0",
                "ppmi1",
                "ppmi2",
                "ppmi3",
                "ppmi5",
                "ppmi6",
                "edsd6",
                "ppmi7",
                "ppmi8",
                "ppmi9",
            ],
            "validation_datasets": ["ppmi_test"],
            "filters": None,
        },
        "parameters": {},
    }
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    # this test only ensures that the algorithm runs smoothly without errors
    assert result
