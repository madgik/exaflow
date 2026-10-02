import math
from unittest.mock import Mock

import pytest

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.algorithm import _values_equal
from exaflow.algorithms.utils.inputdata_utils import Inputdata


class DummyEngine:
    def __init__(self, results):
        self._results = results

    def run_udf(
        self,
        func,
        check_min_rows,
        add_dataset_variable,
        kw_args,
    ):
        return self._results


class DummyAlgorithm(Algorithm):
    def run(self):
        return None


def _make_algorithm(results):
    return DummyAlgorithm(
        engine=DummyEngine(results),
        logger=Mock(),
        inputdata=Inputdata(
            data_model="dummy:0.1",
            datasets=["d1"],
            variables=["x1", "y1"],
        ),
        x=["x1"],
        y=["y1"],
        parameters={},
    )


def _dummy_udf():
    return None


def test_run_local_udf_identical_results_accepts_nan_payloads():
    algo = _make_algorithm(
        [
            {"metric": float("nan"), "nested": {"values": [1.0, float("nan")]}},
            {"metric": float("nan"), "nested": {"values": [1.0, float("nan")]}},
        ]
    )

    result = algo.run_local_udf(_dummy_udf, kw_args={}, identical_results=True)

    assert math.isnan(result["metric"])
    assert math.isnan(result["nested"]["values"][1])


def test_run_local_udf_identical_results_raises_on_real_mismatch():
    algo = _make_algorithm([{"metric": 1.0}, {"metric": 2.0}])

    with pytest.raises(RuntimeError, match="Inconsistent UDF responses"):
        algo.run_local_udf(_dummy_udf, kw_args={}, identical_results=True)

    algo._logger.info.assert_called_once_with(
        "Inconsistent UDF responses for '%s': "
        "worker result at index 0 = %r; worker result at index %s = %r",
        "_dummy_udf",
        {"metric": 1.0},
        1,
        {"metric": 2.0},
    )


@pytest.mark.parametrize(
    "left,right,expected",
    [
        pytest.param(
            1605.7358809905684, 1605.7358809905681, True, id="reproduced-lr-drift"
        ),
        pytest.param(100.0, 100.0000000001, True, id="whole-and-nonwhole-floats"),
        pytest.param(1.5, 1.50000000075, True, id="relative-tolerance"),
        pytest.param(1.5, 1.500000003, False, id="outside-relative-tolerance"),
        pytest.param(0.0, 1e-12, True, id="absolute-tolerance-boundary"),
        pytest.param(
            0.0,
            math.nextafter(1e-12, math.inf),
            False,
            id="outside-absolute-tolerance",
        ),
        pytest.param(0.0, -5e-13, True, id="negative-near-zero"),
        pytest.param(100.0, 100.0, True, id="equal-whole-floats"),
        pytest.param(100.0, 101.0, False, id="different-whole-floats"),
        pytest.param(1_000_000_000.0, 1_000_000_001.0, False, id="large-float-counts"),
        pytest.param(
            10_000_000_000_000_000.0,
            10_000_000_000_000_002.0,
            False,
            id="accepted-tradeoff-large-whole-statistics",
        ),
        pytest.param(100, 100, True, id="equal-integers"),
        pytest.param(100, 101, False, id="different-integers"),
        pytest.param(100, 100.0, True, id="equal-mixed-numeric-types"),
        pytest.param(100, 100.0000000001, False, id="mixed-types-stay-exact"),
        pytest.param(100.0000000001, 100, False, id="mixed-types-reversed"),
        pytest.param(float("nan"), float("nan"), True, id="matching-nans"),
        pytest.param(float("nan"), 0.0, False, id="nan-and-finite"),
        pytest.param(math.inf, math.inf, True, id="matching-positive-infinities"),
        pytest.param(-math.inf, -math.inf, True, id="matching-negative-infinities"),
        pytest.param(math.inf, -math.inf, False, id="opposite-infinities"),
        pytest.param(math.inf, 1.5, False, id="infinity-and-finite"),
        pytest.param("age", "sleep", False, id="different-feature-names"),
        pytest.param(True, False, False, id="different-booleans"),
        pytest.param(None, None, True, id="matching-null-values"),
        pytest.param(None, 0.0, False, id="null-and-number"),
    ],
)
def test_values_equal_numeric_policy(left, right, expected):
    assert _values_equal(left, right) is expected
    assert _values_equal(right, left) is expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        pytest.param(
            {"n_obs": 100, "matrix": [[1.23, 2.34], [3.45, 4.56]]},
            {"matrix": [[1.2300000001, 2.34], [3.45, 4.5600000001]], "n_obs": 100},
            True,
            id="nested-matrix-and-dictionary-order",
        ),
        pytest.param(
            [{"values": (1.23, [2.34])}],
            [{"values": (1.2300000001, [2.3400000001])}],
            True,
            id="lists-dictionaries-tuples-and-lists",
        ),
        pytest.param(
            {"class_count": [1_000_000_000.0]},
            {"class_count": [1_000_000_001.0]},
            False,
            id="nested-whole-float-counts",
        ),
        pytest.param(
            {"coefficients": [100.0]},
            {"coefficients": [100.0000000001]},
            True,
            id="whole-valued-coefficient-with-rounding",
        ),
        pytest.param(
            {"matrix": [[1.23, 2.34]]},
            {"matrix": [[1.23, 2.34001]]},
            False,
            id="nested-significant-float-difference",
        ),
        pytest.param(
            {"values": [1.23], "n_obs": 100},
            {"values": [1.23], "n_obs": 101},
            False,
            id="nested-integer-difference",
        ),
        pytest.param({"a": 1.23}, {"b": 1.23}, False, id="different-keys"),
        pytest.param({"a": 1.23}, {"a": 1.23, "b": 0}, False, id="extra-key"),
        pytest.param([[1.23, 2.34]], [[1.23]], False, id="inner-list-length"),
        pytest.param([[1.23]], [[1.23], []], False, id="outer-list-length"),
        pytest.param([1.23, 2.34], [2.34, 1.23], False, id="element-order"),
        pytest.param({"values": []}, {"values": {}}, False, id="container-mismatch"),
        pytest.param([], [], True, id="empty-lists"),
        pytest.param({}, {}, True, id="empty-dictionaries"),
    ],
)
def test_values_equal_nested_results(left, right, expected):
    assert _values_equal(left, right) is expected
    assert _values_equal(right, left) is expected


def test_run_local_udf_identical_results_accepts_nested_rounding():
    results = [
        {"n_obs": 100, "matrix": [[100.0, 1.23]]},
        {"n_obs": 100, "matrix": [[100.0000000001, 1.2300000001]]},
    ]
    algo = _make_algorithm(results)

    result = algo.run_local_udf(_dummy_udf, kw_args={}, identical_results=True)

    assert result is results[0]
    algo._logger.info.assert_not_called()


def test_run_local_udf_identical_results_logs_nested_mismatch():
    results = [
        {"matrix": [[1.23, 2.34]]},
        {"matrix": [[1.23, 2.34001]]},
    ]
    algo = _make_algorithm(results)

    with pytest.raises(RuntimeError, match="Inconsistent UDF responses"):
        algo.run_local_udf(_dummy_udf, kw_args={}, identical_results=True)

    algo._logger.info.assert_called_once_with(
        "Inconsistent UDF responses for '%s': "
        "worker result at index 0 = %r; worker result at index %s = %r",
        "_dummy_udf",
        results[0],
        1,
        results[1],
    )
