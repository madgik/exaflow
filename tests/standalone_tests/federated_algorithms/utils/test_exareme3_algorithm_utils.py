import math

import pytest

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
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
        inputdata=Inputdata(
            data_model="dummy:0.1",
            datasets=["d1"],
            x=["x1"],
            y=["y1"],
        ),
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
