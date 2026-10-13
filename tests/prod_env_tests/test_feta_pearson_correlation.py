"""Compare both Pearson implementations through a deployed /analysis API."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import analysis_request
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

CASES = get_test_params(
    Path(__file__).parent / "expected" / "pearson_correlation_expected.json",
)


def assert_identical_structure_and_values(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_identical_structure_and_values(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_identical_structure_and_values(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )
    else:
        assert actual == expected


@pytest.mark.parametrize("test_input, _expected", CASES)
@pytest.mark.parametrize("omitted_x", [False, True])
def test_feta_pearson_matches_normal_pearson(test_input, _expected, omitted_x):
    payload = deepcopy(test_input)
    payload.pop("preprocessing", None)
    if omitted_x:
        payload["algorithm"]["x"] = None
    columns = list(
        dict.fromkeys((payload["algorithm"].get("x") or []) + payload["algorithm"]["y"])
    )
    rules = [
        {"id": column, "type": "double", "operator": "is_not_null", "value": None}
        for column in columns
    ]
    if payload["inputdata"].get("filters"):
        rules.append(payload["inputdata"]["filters"])
    payload["inputdata"]["filters"] = {"condition": "AND", "rules": rules}
    # The same explicit complete-case selection goes to both implementations.
    # Normal Pearson requires its missing-value handler; after filtering it is
    # a no-op. FETA deliberately rejects preprocessing and missing values.
    normal = parse_response(analysis_request("pearson_correlation", payload))
    feta = parse_response(
        analysis_request("feta_pearson_correlation", payload, drop_na=False)
    )
    assert_identical_structure_and_values(feta, normal)
