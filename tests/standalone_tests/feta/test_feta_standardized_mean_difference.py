import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.statistics.standardized_mean_difference import (
    FederatedStandardizedMeanDifference,
)
from exaflow.algorithms.feta import ColumnMetadata
from exaflow.algorithms.feta.feta_standardized_mean_difference import (
    FetaStandardizedMeanDifference,
)
from exaflow.feta.inputdata import Inputdata
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_anova_oneway import CentralAggregation
from tests.standalone_tests.feta.test_feta_anova_oneway import assert_nested


def inputs(runtime, data):
    integer = data.group.dtype.kind in "iu"
    return runtime.initialize(
        Inputdata(data_model="test:1.0", datasets=["a"], variables=["group", "value"]),
        {"x": ["group"], "y": ["value"]},
        {"x"},
        {
            "group": ColumnMetadata(
                "int" if integer else "text", True, tuple(data.group.unique().tolist())
            )
        },
    )


@pytest.mark.parametrize("labels", [["λ", "a", "long"], [2, 10, 1]])
@pytest.mark.parametrize("sizes", [[60], [15, 20, 25], [0, 28, 32]])
@pytest.mark.parametrize("minimum", [0, 18, 25])
def test_smd_parity(labels, sizes, minimum):
    rng = np.random.default_rng(57)
    data = pd.DataFrame({"group": np.resize(labels, 60), "value": rng.normal(size=60)})
    edges = np.cumsum([0, *sizes])
    parts = [data.iloc[a:b] for a, b in zip(edges[:-1], edges[1:])]
    eligible = pd.concat([part for part in parts if len(part) >= minimum])
    native = (
        FederatedStandardizedMeanDifference(CentralAggregation())
        .pairwise_by_group(
            data=eligible,
            value_var="value",
            group_var="group",
            minimum_group_size=minimum,
        )[["group1", "group2", "smd"]]
        .to_dict(orient="records")
    )
    runtime, workers, aggregation = federation(parts, minimum_rows=minimum)
    with runtime:
        actual = FetaStandardizedMeanDifference().run(**inputs(runtime, data))
        assert_nested(actual.model_dump(), {"comparisons": native})
        assert runtime.minimum_group_size == minimum
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


def test_strictest_participating_threshold_excludes_skipped_worker():
    parts = [
        pd.DataFrame({"group": ["a", "b"] * 6, "value": np.arange(12.0)})
        for _ in range(3)
    ]
    runtime, workers, _ = federation(parts)
    for worker, threshold in zip(workers, [2, 15, 9]):
        worker.service.minimum_rows = threshold
    with runtime:
        data = pd.concat(parts)
        result = FetaStandardizedMeanDifference().run(**inputs(runtime, data))
        assert runtime.minimum_group_size == 9
        assert len(runtime.workers) == 2
        assert len(result.comparisons) == 1


@pytest.mark.parametrize("labels", [["a"] * 8, ["a"] * 4 + ["b"] * 4])
def test_constant_groups_and_no_pairs(labels):
    data = pd.DataFrame({"group": labels, "value": np.ones(8)})
    runtime, _, _ = federation([data])
    with runtime:
        result = FetaStandardizedMeanDifference().run(**inputs(runtime, data))
        assert all(comparison.smd == 0.0 for comparison in result.comparisons)
        assert len(result.comparisons) == (0 if len(set(labels)) == 1 else 1)


def test_strictest_threshold_filters_global_groups():
    data = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 8, "value": np.arange(12.0)})
    runtime, workers, _ = federation([data, data])
    workers[0].service.minimum_rows = 2
    workers[1].service.minimum_rows = 9
    with runtime:
        result = FetaStandardizedMeanDifference().run(**inputs(runtime, data))
        assert runtime.minimum_group_size == 9
        assert result.comparisons == []  # Global group a has eight observations.


def test_absent_privacy_report_is_rejected_and_cleaned(monkeypatch):
    data = pd.DataFrame({"group": ["a", "b"] * 3, "value": np.arange(6.0)})
    runtime, workers, _ = federation([data])
    original = workers[0].initialize

    def initialize(request):
        response = original(request)
        response.ClearField("minimum_group_size")
        return response

    monkeypatch.setattr(workers[0], "initialize", initialize)
    with pytest.raises(ValueError, match="did not report its privacy minimum"):
        with runtime:
            inputs(runtime, data)
    assert not workers[0].service.store.contexts
