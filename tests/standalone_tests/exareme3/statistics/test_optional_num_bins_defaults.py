from unittest.mock import Mock

import pytest

from exaflow.algorithms.exareme3.statistics.binned_mann_whitney_u_test import (
    BinnedMannWhitneyUTest,
)
from exaflow.algorithms.exareme3.statistics.quartiles import QuartilesAlgorithm
from exaflow.algorithms.utils.inputdata_utils import Inputdata


def _inputdata():
    return Inputdata(
        data_model="dementia:0.1",
        datasets=["dataset_a"],
        variables=["age", "group"],
    )


def _quartiles(parameters):
    return QuartilesAlgorithm(
        engine=Mock(),
        logger=Mock(),
        inputdata=_inputdata(),
        y=["age"],
        metadata={"age": {"sql_type": "int"}},
        parameters=parameters,
    )


def _mann_whitney(parameters):
    return BinnedMannWhitneyUTest(
        engine=Mock(),
        logger=Mock(),
        inputdata=_inputdata(),
        x=["group"],
        y=["age"],
        parameters=parameters,
    )


def _bins_passed_to_udf(algorithm, payload):
    captured = {}

    def _run_local_udf(func, kw_args, *, identical_results=False):
        captured["num_bins"] = kw_args["num_bins"]
        return payload

    algorithm.run_local_udf = _run_local_udf
    algorithm.run()
    return captured["num_bins"]


_QUARTILE_PAYLOAD = {
    "quantiles": [
        {"q": 0.25, "value": 1.0, "actual_q": None},
        {"q": 0.5, "value": 2.0, "actual_q": None},
        {"q": 0.75, "value": 3.0, "actual_q": None},
    ]
}

_MANN_WHITNEY_PAYLOAD = {
    "u_stat": 1.0,
    "p_value": 0.5,
    "z_score": 0.0,
    "n1": 3,
    "n2": 3,
}

_MANN_WHITNEY_PARAMETERS = {
    "alt_hypothesis": "two-sided",
    "groupA": "a",
    "groupB": "b",
}


@pytest.mark.parametrize("parameters", [{}, {"num_bins": None}])
def test_quartiles_uses_default_bins_when_num_bins_absent_or_null(parameters):
    default_bins = QuartilesAlgorithm.get_specification().parameters["num_bins"].default

    assert default_bins == 20
    num_bins = _bins_passed_to_udf(_quartiles(parameters), _QUARTILE_PAYLOAD)
    assert num_bins == default_bins


def test_quartiles_uses_explicit_num_bins():
    num_bins = _bins_passed_to_udf(_quartiles({"num_bins": 8}), _QUARTILE_PAYLOAD)

    assert num_bins == 8


@pytest.mark.parametrize("num_bins", [None, "absent"])
def test_binned_mann_whitney_uses_default_bins_when_num_bins_absent_or_null(num_bins):
    parameters = dict(_MANN_WHITNEY_PARAMETERS)
    if num_bins is None:
        parameters["num_bins"] = None
    default_bins = (
        BinnedMannWhitneyUTest.get_specification().parameters["num_bins"].default
    )

    assert default_bins == 40
    assert (
        _bins_passed_to_udf(_mann_whitney(parameters), _MANN_WHITNEY_PAYLOAD)
        == default_bins
    )


def test_binned_mann_whitney_uses_explicit_num_bins():
    parameters = dict(_MANN_WHITNEY_PARAMETERS)
    parameters["num_bins"] = 12

    assert _bins_passed_to_udf(_mann_whitney(parameters), _MANN_WHITNEY_PAYLOAD) == 12
