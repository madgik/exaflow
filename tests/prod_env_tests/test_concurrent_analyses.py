from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import analysis_request
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

EXPECTED_DIR = Path(__file__).parent / "expected"
CASES_PER_ALGORITHM = 4


def _requests():
    return [
        (algorithm, test_input)
        for algorithm in ["linear_regression", "pearson_correlation"]
        for test_input, _ in get_test_params(
            EXPECTED_DIR / f"{algorithm}_expected.json",
            slc=slice(CASES_PER_ALGORITHM),
        )
    ]


def _run(request):
    return parse_response(analysis_request(*request))


def _assert_same(actual, expected, path="result"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_same(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        # Aggregation order differs between runs, so allow rounding noise.
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9, nan_ok=True), path
    else:
        assert actual == expected, path


def test_concurrent_analyses_match_serial_results():
    """Analyses running at the same time must not share or mix aggregation state."""
    requests = _requests()
    serial = [_run(request) for request in requests]

    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        concurrent = list(pool.map(_run, requests))

    for (algorithm, _), actual, expected in zip(requests, concurrent, serial):
        _assert_same(actual, expected, algorithm)
