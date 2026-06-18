import pytest

from tests.algorithm_validation_tests.flower.helpers import analysis_request
from tests.algorithm_validation_tests.flower.helpers import parse_response


@pytest.fixture(scope="class")
def get_algorithm_result():
    cache = {}

    def _get_algorithm_result(algname, test_input):
        test_case_num = test_input["test_case_num"]
        key = (algname, test_case_num)
        if key not in cache:
            response = analysis_request(algname, test_input)
            result = parse_response(response)
            cache[key] = result
        return cache[key]

    return _get_algorithm_result
