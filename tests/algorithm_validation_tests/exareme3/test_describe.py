from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme3.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme3.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme3.helpers import get_test_params
from tests.algorithm_validation_tests.exareme3.helpers import parse_response

algorithm_name = "describe"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_describe(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    compare_results(result, expected)


def compare_results(result, expected):
    # sort records by variable and dataset in order to compare them
    varbased_res = sorted(
        result["variable_based"],
        key=lambda rec: rec["variable"] + rec["dataset"],
    )
    varbased_exp = sorted(
        expected["variable_based"],
        key=lambda rec: rec["variable"] + rec["dataset"],
    )
    compare_records(varbased_res, varbased_exp)

    modbased_res = sorted(
        result["model_based"],
        key=lambda rec: rec["variable"] + rec["dataset"],
    )
    modbased_exp = sorted(
        expected["model_based"],
        key=lambda rec: rec["variable"] + rec["dataset"],
    )
    compare_records(modbased_res, modbased_exp)


def compare_records(records1, records2):
    for rec1, rec2 in zip(records1, records2):
        assert rec1["variable"] == rec2["variable"]
        assert rec1["dataset"] == rec2["dataset"]
        data1, data2 = rec1["data"], rec2["data"]
        if not data1 and not data2:
            pass
        elif data1 and data2:
            if "mean" in data1:
                compare_numerical_data(data1, data2)
            if "counts" in data1:
                compare_nominal_data(data1, data2)
        else:
            pytest.fail(f"data1 and data2 are different: {data1} != {data2}")


def compare_numerical_data(data1, data2):
    assert data1["num_dtps"] == data2["num_dtps"]
    assert data1["num_na"] == data2["num_na"]
    assert data1["num_total"] == data2["num_total"]
    assert_allclose(data1["mean"], data2["mean"])
    if data1["std"] is not None and data2["std"] is not None:
        assert_allclose(data1["std"], data2["std"])
    assert_allclose(data1["min"], data2["min"])
    assert_allclose(data1["max"], data2["max"])
    assert (data1["q1"] and data2["q1"]) or (not data1["q1"] and not data2["q1"])
    assert (data1["q2"] and data2["q2"]) or (not data1["q2"] and not data2["q2"])
    assert (data1["q3"] and data2["q3"]) or (not data1["q3"] and not data2["q3"])
    if data1["q1"]:
        assert_allclose(data1["q1"], data2["q1"])
        assert_allclose(data1["q2"], data2["q2"])
        assert_allclose(data1["q3"], data2["q3"])


def compare_nominal_data(data1, data2):
    assert data1["num_dtps"] == data2["num_dtps"]
    assert data1["num_na"] == data2["num_na"]
    assert data1["num_total"] == data2["num_total"]
    assert data1["counts"] == data2["counts"]
