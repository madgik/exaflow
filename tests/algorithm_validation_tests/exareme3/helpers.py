import functools
import json

import numpy as np
import pytest
import requests

MISSING_VALUES_STEP_NAME = "missing_values_handler"
MISSING_VALUE_DROP_STRATEGY = "drop"


def algorithm_request(algorithm: str, input: dict, drop_na: bool = True):
    request_payload = dict(input)
    if drop_na:
        preprocessing = dict(request_payload.get("preprocessing") or {})
        if MISSING_VALUES_STEP_NAME not in preprocessing:
            strategies = _build_default_drop_strategies(request_payload)
            if strategies:
                preprocessing[MISSING_VALUES_STEP_NAME] = {"strategies": strategies}
        request_payload["preprocessing"] = preprocessing

    url = "http://127.0.0.1:5100/algorithms" + f"/{algorithm}"
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=json.dumps(request_payload), headers=headers)
    return response


def _build_default_drop_strategies(request_payload: dict) -> dict:
    inputdata = request_payload.get("inputdata") or {}
    x_vars = inputdata.get("x") or []
    y_vars = inputdata.get("y") or []
    variables = list(dict.fromkeys(list(x_vars) + list(y_vars)))
    return {var: MISSING_VALUE_DROP_STRATEGY for var in variables}


def parse_response(response) -> dict:
    if response.status_code != 200:
        msg = f"Unexpected response status: '{response.status_code}'. "
        msg += f"Response message: '{response.content}'"
        raise ValueError(msg)
    try:
        result = json.loads(response.content)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"The result is not valid json:\n{response.content}") from None
    return result


def get_test_params(expected_file, slc=None, skip_indices=None, skip_reason=None):
    """
    Gets parameters for algorithm validation tests from expected file

    Can get the whole list present in the expected file or a given slice. Can
    also skip some tests based on their indices.

    Parameters
    ----------
    expected_file : pathlib.Path
        File in json format containing a list of test cases, where a test case
        is a pair of input/output for a given algorithm
    slc : slice | None
        If not None it gets only the given slice
    skip_indices : list[int] | None
        Indices of tests to skip
    skip_reason : str | None
        Reason for skipping tests, combine with previous parameter
    """
    with expected_file.open() as f:
        params = json.load(f)["test_cases"]
    slc = slc or slice(len(params))
    params = [(p["input"], p["output"]) for p in params[slc]]

    def skip(*param):
        return pytest.param(*param, marks=pytest.mark.skip(reason=skip_reason))

    if skip_indices:
        params = [skip(*p) if i in skip_indices else p for i, p in enumerate(params)]
    return params


assert_allclose = functools.partial(
    np.testing.assert_allclose,
    rtol=1e-6,
    atol=1e-9,
)
