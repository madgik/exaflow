"""Compare both paired t-test runtimes through a deployed analysis API."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import analysis_request
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response


@pytest.mark.parametrize(
    "test_input, _expected",
    get_test_params(Path(__file__).parent / "expected" / "ttest_paired_expected.json"),
)
def test_feta_paired_matches_normal(test_input, _expected):
    payload = deepcopy(test_input)
    payload.pop("preprocessing", None)
    columns = list(dict.fromkeys(payload["algorithm"]["x"] + payload["algorithm"]["y"]))
    rules = [
        {"id": column, "type": "double", "operator": "is_not_null", "value": None}
        for column in columns
    ]
    if payload["inputdata"].get("filters"):
        rules.append(payload["inputdata"]["filters"])
    payload["inputdata"]["filters"] = {"condition": "AND", "rules": rules}
    normal = parse_response(analysis_request("ttest_paired", payload))
    feta = parse_response(analysis_request("feta_ttest_paired", payload, drop_na=False))
    assert feta.keys() == normal.keys()
    for key, expected in normal.items():
        if expected is None or isinstance(expected, str):
            assert feta[key] == expected
        else:
            np.testing.assert_allclose(feta[key], expected, rtol=1e-10, atol=1e-12)
