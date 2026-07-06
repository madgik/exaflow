import json
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.conftest import analysis_request

input_file = Path(__file__).parent / "input" / "kmeans_input.json"


def get_test_inputs():
    with input_file.open() as f:
        return json.load(f)["test_cases"]


@pytest.mark.parametrize("test_input", get_test_inputs())
def test_kmeans(test_input):
    response = analysis_request("kmeans", test_input)
    try:
        result = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"The result is not valid json:\n{response.text}") from None

    # this test only ensures that the algorithm runs smoothly without errors
    assert result
