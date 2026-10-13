from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import ols

from exaflow.algorithms.exareme3.statistics.anova_twoway import AnovaTwoWay
from exaflow.algorithms.feta.feta_anova_twoway import FetaAnovaTwoWay
from exaflow.feta.inputdata import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_anova_oneway import CentralAggregation
from tests.standalone_tests.feta.test_feta_anova_oneway import assert_nested


def reference(frame, sstype):
    proxy = SimpleNamespace(
        x=["a", "b"],
        y=["value"],
        get_parameter=lambda name: sstype,
        run_local_udf=lambda func, kw_args, identical_results: func(
            agg_client=CentralAggregation(), data=frame, **kw_args
        ),
    )
    return AnovaTwoWay.run(proxy).model_dump()


def selection():
    return Inputdata(
        data_model="test:1.0", datasets=["a"], variables=["a", "b", "value"]
    )


def data():
    rng = np.random.default_rng(91)
    a = np.resize(["α", "long label", "z"], 60)
    b = np.repeat(["control", "treated"], 30)
    return pd.DataFrame(
        {
            "a": a,
            "b": b,
            "value": rng.normal(size=60) + (a == "α") * 0.4 + (b == "treated") * 0.7,
        }
    )


@pytest.mark.parametrize("sstype", [1, 2])
@pytest.mark.parametrize("sizes", [[60], [13, 22, 25], [0, 29, 31]])
@pytest.mark.parametrize("minimum_rows", [0, 14])
def test_two_way_parity(sstype, sizes, minimum_rows):
    frame = data()
    edges = np.cumsum([0, *sizes])
    parts = [frame.iloc[a:b] for a, b in zip(edges[:-1], edges[1:])]
    eligible = pd.concat([part for part in parts if len(part) >= minimum_rows])
    runtime, workers, aggregation = federation(parts, minimum_rows=minimum_rows)
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["a", "b"], "y": ["value"]})
        result = FetaAnovaTwoWay(
            x_names=["a", "b"], y_names=["value"], parameters={"sstype": sstype}
        ).run(**tensors)
        assert_nested(result.model_dump(), reference(eligible, sstype))
        table = sm.stats.anova_lm(
            ols("value ~ C(a) * C(b)", data=eligible).fit(), typ=sstype
        )
        np.testing.assert_allclose(
            result.sum_sq, table["sum_sq"], rtol=1e-10, atol=1e-12
        )
        np.testing.assert_array_equal(result.df, table["df"].astype(int))
        np.testing.assert_allclose(
            result.f_stat[:3], table["F"][:3], rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            result.f_pvalue[:3], table["PR(>F)"][:3], rtol=1e-10, atol=1e-12
        )
        assert runtime.step == 18  # two unions, count, and three reductions per model
    assert not aggregation.rounds
    assert all(not w.service.store.contexts for w in workers)


@pytest.mark.parametrize("sstype", [1, 2])
def test_rank_deficient_missing_cells(sstype):
    frame = data()
    frame = frame[~((frame.a == "z") & (frame.b == "treated"))]
    runtime, _, _ = federation([frame.iloc[:20], frame.iloc[20:]])
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["a", "b"], "y": ["value"]})
        result = FetaAnovaTwoWay(
            x_names=["a", "b"], y_names=["value"], parameters={"sstype": sstype}
        ).run(**tensors)
        assert_nested(result.model_dump(), reference(frame, sstype))


@pytest.mark.parametrize("column", ["a", "b"])
def test_single_factor_level_errors_and_cleanup(column):
    frame = data()
    frame[column] = "single"
    runtime, workers, _ = federation([frame])
    with pytest.raises(BadUserInput, match=f"variable {column} has less than 2 levels"):
        with runtime:
            tensors = runtime.initialize(selection(), {"x": ["a", "b"], "y": ["value"]})
            FetaAnovaTwoWay(x_names=["a", "b"], y_names=["value"]).run(**tensors)
    assert all(not w.service.store.contexts for w in workers)


@pytest.mark.parametrize("sstype", [1, 2])
def test_saturated_design(sstype):
    frame = pd.DataFrame(
        {
            "a": ["a", "a", "b", "b"],
            "b": ["a", "b", "a", "b"],
            "value": [1.0, 3.0, 2.0, 7.0],
        }
    )
    runtime, _, _ = federation([frame])
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["a", "b"], "y": ["value"]})
        result = FetaAnovaTwoWay(
            x_names=["a", "b"], y_names=["value"], parameters={"sstype": sstype}
        ).run(**tensors)
        assert_nested(result.model_dump(), reference(frame, sstype))
        assert result.f_stat == [None] * 4


def test_type_one_respects_requested_factor_order():
    frame = data()
    runtime, _, _ = federation([frame.iloc[:23], frame.iloc[23:]])
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["b", "a"], "y": ["value"]})
        result = FetaAnovaTwoWay(
            x_names=["b", "a"], y_names=["value"], parameters={"sstype": 1}
        ).run(**tensors)
        expected = reference(frame.rename(columns={"a": "b", "b": "a"}), 1)
        expected["terms"] = ["b", "a", "b:a", "Residuals"]
        assert_nested(result.model_dump(), expected)
