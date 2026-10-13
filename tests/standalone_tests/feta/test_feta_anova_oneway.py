from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from exaflow.algorithms.exareme3.statistics.anova_oneway import AnovaOneWay
from exaflow.algorithms.feta.feta_anova_oneway import FetaAnovaOneWay
from exaflow.feta.inputdata import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation


class CentralAggregation:
    def sum(self, value):
        return value

    def min(self, value):
        return value

    def max(self, value):
        return value

    def union(self, value):
        return np.unique(value)


def reference(frame):
    proxy = SimpleNamespace(
        x=["group"],
        y=["value"],
        run_local_udf=lambda func, kw_args, identical_results: func(
            agg_client=CentralAggregation(), data=frame, **kw_args
        ),
    )
    return AnovaOneWay.run(proxy).model_dump()


def assert_nested(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested(left, right)
    elif isinstance(expected, str) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True
        )


def selection():
    return Inputdata(
        data_model="test:1.0", datasets=["a"], variables=["group", "value"]
    )


@pytest.mark.parametrize("sizes", [[48], [13, 16, 19], [0, 27, 21]])
@pytest.mark.parametrize("minimum_rows", [0, 14])
@pytest.mark.parametrize(
    "categories", [["control", "treated"], ["α", "long category", "z"]]
)
def test_anova_parity(sizes, minimum_rows, categories):
    rng = np.random.default_rng(72)
    frame = pd.DataFrame(
        {
            "group": np.resize(categories, sum(sizes)),
            "value": rng.normal(size=sum(sizes)),
        }
    )
    edges = np.cumsum([0, *sizes])
    parts = [frame.iloc[a:b] for a, b in zip(edges[:-1], edges[1:])]
    eligible = pd.concat([part for part in parts if len(part) >= minimum_rows])
    runtime, workers, aggregation = federation(parts, minimum_rows=minimum_rows)
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["group"], "y": ["value"]})
        result = FetaAnovaOneWay(x_names=["group"], y_names=["value"]).run(**tensors)
        assert_nested(result.model_dump(), reference(eligible))
        centralized = stats.f_oneway(
            *[eligible.value[eligible.group == c] for c in categories]
        )
        np.testing.assert_allclose(
            [result.anova_table["f_stat"], result.anova_table["p_value"]],
            [centralized.statistic, centralized.pvalue],
            atol=1e-12,
        )
        assert runtime.step == 1 + 5 * len(categories)
    assert not aggregation.rounds
    assert all(not w.service.store.contexts for w in workers)


def test_groups_on_separate_workers_have_neutral_extrema():
    a = pd.DataFrame({"group": ["a"] * 12, "value": np.arange(12, dtype=float)})
    b = pd.DataFrame({"group": ["b"] * 13, "value": np.arange(13, dtype=float) + 3})
    runtime, _, _ = federation([a, b], minimum_rows=10)
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["group"], "y": ["value"]})
        result = FetaAnovaOneWay(x_names=["group"], y_names=["value"]).run(**tensors)
        assert_nested(result.model_dump(), reference(pd.concat([a, b])))
        assert result.min_max_per_group == {
            "categories": ["a", "b"],
            "min": [0.0, 3.0],
            "max": [11.0, 15.0],
        }


@pytest.mark.parametrize("groups", [[], ["a"] * 8])
def test_insufficient_levels(groups):
    frame = pd.DataFrame(
        {
            "group": pd.Series(groups, dtype=object),
            "value": np.arange(len(groups), dtype=float),
        }
    )
    runtime, workers, _ = federation([frame])
    with pytest.raises(BadUserInput, match="only one level"):
        with runtime:
            tensors = runtime.initialize(selection(), {"x": ["group"], "y": ["value"]})
            FetaAnovaOneWay(x_names=["group"], y_names=["value"]).run(**tensors)
    assert all(not w.service.store.contexts for w in workers)


def test_constant_sample_preserves_existing_behavior():
    frame = pd.DataFrame({"group": ["a"] * 12 + ["b"] * 12, "value": np.ones(24)})
    runtime, _, _ = federation([frame])
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["group"], "y": ["value"]})
        result = FetaAnovaOneWay(x_names=["group"], y_names=["value"]).run(**tensors)
        assert_nested(result.model_dump(), reference(frame))


def test_skipped_workers_do_not_contribute_categories():
    skipped = pd.DataFrame({"group": ["rare"] * 3, "value": [1.0, 2.0, 3.0]})
    eligible = pd.DataFrame(
        {"group": ["a"] * 12 + ["b"] * 12, "value": np.arange(24, dtype=float)}
    )
    runtime, _, _ = federation([skipped, eligible], minimum_rows=10)
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["group"], "y": ["value"]})
        result = FetaAnovaOneWay(x_names=["group"], y_names=["value"]).run(**tensors)
        assert_nested(result.model_dump(), reference(eligible))
        assert result.min_max_per_group["categories"] == ["a", "b"]
