"""Compare complete ANOVA responses through a development-deployed analysis API."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import analysis_request
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response


def assert_response(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_response(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_response(left, right)
    elif expected is None or isinstance(expected, str):
        assert actual == expected
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )


@pytest.mark.parametrize(
    "test_input, _expected",
    get_test_params(Path(__file__).parent / "expected" / "anova_oneway_expected.json"),
)
def test_feta_anova_matches_normal(test_input, _expected):
    payload = deepcopy(test_input)
    payload.pop("preprocessing", None)
    columns = list(dict.fromkeys(payload["algorithm"]["x"] + payload["algorithm"]["y"]))
    rules = [
        {
            "id": column,
            "type": "string" if column in payload["algorithm"]["x"] else "double",
            "operator": "is_not_null",
            "value": None,
        }
        for column in columns
    ]
    if payload["inputdata"].get("filters"):
        rules.append(payload["inputdata"]["filters"])
    payload["inputdata"]["filters"] = {"condition": "AND", "rules": rules}
    normal = parse_response(analysis_request("anova_oneway", payload))
    feta = parse_response(analysis_request("feta_anova_oneway", payload, drop_na=False))
    assert_response(feta, normal)
