import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from exaflow.algorithms.exareme3.statistics.fisher_exact import fisher_exact_local_step
from exaflow.algorithms.feta.feta_fisher_exact import FetaFisherExact
from exaflow.feta.inputdata import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_anova_oneway import CentralAggregation
from tests.standalone_tests.feta.test_feta_anova_oneway import assert_nested


def inputs(runtime):
    return runtime.initialize(
        Inputdata(
            data_model="test:1.0", datasets=["a"], variables=["factor", "outcome"]
        ),
        {"x": ["factor"], "y": ["outcome"]},
    )


@pytest.mark.parametrize(
    "table", [[[1, 9], [11, 3]], [[0, 10], [6, 4]], [[6, 0], [0, 8]]]
)
@pytest.mark.parametrize("partitions", [1, 3])
def test_fisher_parity(table, partitions):
    data = pd.DataFrame(
        [
            (label, outcome)
            for label, row in zip(["λ", "a"], table)
            for outcome, count in zip(["yes", "no"], row)
            for _ in range(count)
        ],
        columns=["factor", "outcome"],
    )
    parts = [
        data.iloc[indices]
        for indices in np.array_split(np.arange(len(data)), partitions)
    ]
    native = fisher_exact_local_step(
        CentralAggregation(), data, {}, "factor", "outcome"
    )
    runtime, workers, aggregation = federation(parts)
    with runtime:
        result = FetaFisherExact(x_names=["factor"], y_names=["outcome"]).run(
            **inputs(runtime)
        )
        assert_nested(
            result.model_dump(),
            dict(
                odds_ratio=native[0],
                p_value=native[1],
                x_labels=native[2],
                y_labels=native[3],
            ),
        )
        expected = fisher_exact(pd.crosstab(data.factor, data.outcome))
        np.testing.assert_allclose([result.odds_ratio, result.p_value], expected)
        assert sum(event[0] == "prepare" for event in aggregation.events) == 3
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize("levels", [1, 3])
def test_invalid_observed_levels_and_cleanup(levels):
    data = pd.DataFrame(
        {
            "factor": np.resize([str(i) for i in range(levels)], 12),
            "outcome": np.resize(["yes", "no"], 12),
        }
    )
    runtime, workers, aggregation = federation([data])
    with pytest.raises(BadUserInput, match=f"factor 'factor' has {levels}"):
        with runtime:
            FetaFisherExact(x_names=["factor"], y_names=["outcome"]).run(
                **inputs(runtime)
            )
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


def test_privacy_skip_changes_observed_levels():
    data = pd.DataFrame(
        {
            "factor": ["excluded"] * 2 + ["a", "b"] * 6,
            "outcome": ["no"] * 2 + ["yes", "no", "no", "yes"] * 3,
        }
    )
    runtime, workers, aggregation = federation(
        [data.iloc[:2], data.iloc[2:]], minimum_rows=3
    )
    with runtime:
        result = FetaFisherExact(x_names=["factor"], y_names=["outcome"]).run(
            **inputs(runtime)
        )
        assert result.x_labels == ["a", "b"]
        assert result.odds_ratio == 1
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)
