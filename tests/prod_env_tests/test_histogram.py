import json
import re
from pathlib import Path

import pytest
import requests

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request
from tests.algorithm_validation_tests.exareme3.conftest import parse_response
from tests.algorithm_validation_tests.exareme3.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.prod_env_tests import analysis_url

algorithm_name = "histogram"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


def _index_by_identity(histograms):
    """Index a histogram list by (var, grouping_var, grouping_enum) for
    order-agnostic comparison."""
    return {
        (item["var"], item["grouping_var"], item["grouping_enum"]): item
        for item in histograms
    }


def _bins_are_numerical(bins):
    return all(isinstance(b, (int, float)) for b in bins)


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_histogram(test_input, expected):
    # The expected fixtures were generated with equal-width binning.
    if test_input["algorithm"].get("parameters") is None:
        test_input["algorithm"]["parameters"] = {}
    test_input["algorithm"]["parameters"]["histogram_type"] = "simple"

    response = analysis_request(algorithm_name, test_input)
    result = parse_response(response)

    expected_histograms = expected["histogram"]
    actual_histograms = result["histogram"]
    assert len(actual_histograms) == len(expected_histograms)

    actual_by_id = _index_by_identity(actual_histograms)
    expected_by_id = _index_by_identity(expected_histograms)
    assert actual_by_id.keys() == expected_by_id.keys()

    for identity, exp in expected_by_id.items():
        got = actual_by_id[identity]
        # Counts are ints (or None for privacy-masked) — exact equality.
        assert got["counts"] == exp["counts"], identity
        # Bin edges are floats for numerical histograms — allow tolerance;
        # category labels are strings — exact equality.
        if _bins_are_numerical(exp["bins"]):
            assert_allclose(got["bins"], exp["bins"])
        else:
            assert got["bins"] == exp["bins"], identity


def test_histogram_insufficient_data_after_filters_returns_461():
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": ["edsd0"],
            "variables": ["lefthippocampus"],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "condition": "OR",
                        "rules": [
                            {
                                "id": "subjectage",
                                "field": "subjectage",
                                "type": "real",
                                "input": "number",
                                "operator": "greater",
                                "value": 200.0,
                            }
                        ],
                    }
                ],
                "valid": True,
            },
        },
        "preprocessing": [
            {
                "name": "missing_values_handler",
                "parameters": {"strategies": {"lefthippocampus": "drop"}},
            }
        ],
        "algorithm": {
            "name": "histogram",
            "x": None,
            "y": ["lefthippocampus"],
            "parameters": {"bins": 20},
        },
    }
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        analysis_url, data=json.dumps(request_dict), headers=headers
    )

    assert response.status_code == 461, f"Response message: {response.text}"
    assert re.search(
        "The algorithm could not run with the input provided because there are insufficient data.",
        response.text,
    )
