import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from exaflow.algorithms.federated.statistics.chi_squared import ChiSquared
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from exaflow.algorithms.feta import ColumnMetadata
from exaflow.algorithms.feta.feta_chi_squared import FetaChiSquared
from exaflow.feta.inputdata import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_anova_oneway import CentralAggregation
from tests.standalone_tests.feta.test_feta_anova_oneway import assert_nested


def inputs(runtime, categories):
    return runtime.initialize(
        Inputdata(
            data_model="test:1.0", datasets=["a"], variables=["factor", "outcome"]
        ),
        {"x": ["factor"], "y": ["outcome"]},
        {"x", "y"},
        {
            name: ColumnMetadata("text", True, tuple(codes))
            for name, codes in zip(["factor", "outcome"], categories)
        },
    )


@pytest.mark.parametrize("sizes", [[72], [11, 29, 32], [0, 35, 37]])
@pytest.mark.parametrize("minimum_rows", [0, 12])
@pytest.mark.parametrize(
    "categories", [([], []), (["λ", "a", "absent", "z"], ["yes", "no", "absent"])]
)
def test_chi_squared_parity(sizes, minimum_rows, categories):
    rng = np.random.default_rng(72)
    data = pd.DataFrame(
        {
            "factor": np.resize(["a", "z", "λ"], 72),
            "outcome": rng.choice(["yes", "no"], 72),
        }
    )
    edges = np.cumsum([0, *sizes])
    parts = [data.iloc[a:b] for a, b in zip(edges[:-1], edges[1:])]
    eligible = pd.concat([part for part in parts if len(part) >= minimum_rows])
    native = ChiSquared(NumpyAggregator(CentralAggregation())).compute(
        eligible,
        "factor",
        "outcome",
        factor_categories=categories[0],
        outcome_categories=categories[1],
        dropna=True,
    )
    expected = dict(
        chi2=native[0],
        p_value=native[1],
        dof=native[2],
        expected=native[3].tolist(),
        x_labels=native.x_labels,
        y_labels=native.y_labels,
    )
    runtime, workers, aggregation = federation(parts, minimum_rows=minimum_rows)
    with runtime:
        result = FetaChiSquared().run(**inputs(runtime, categories)).model_dump()
        assert_nested(result, expected)
        assert sum(event[0] == "prepare" for event in aggregation.events) == (
            3 if not categories[0] else 1
        )
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize("table", [[[1, 9], [11, 3]], [[6, 7]]])
def test_continuity_correction_and_single_level(table):
    data = pd.DataFrame(
        [
            (str(i), str(j))
            for i, row in enumerate(table)
            for j, count in enumerate(row)
            for _ in range(count)
        ],
        columns=["factor", "outcome"],
    )
    runtime, _, _ = federation([data])
    with runtime:
        result = FetaChiSquared().run(**inputs(runtime, ([], [])))
        chi2, p, dof, expected = chi2_contingency(table)
        np.testing.assert_allclose(
            [result.chi2, result.p_value, result.dof], [chi2, p, dof]
        )
        np.testing.assert_allclose(result.expected, expected)


def test_empty_table_error_and_cleanup():
    data = pd.DataFrame({"factor": ["unknown"] * 4, "outcome": ["yes"] * 4})
    runtime, workers, aggregation = federation([data])
    with pytest.raises(BadUserInput, match="empty after pruning"):
        with runtime:
            FetaChiSquared().run(**inputs(runtime, (["known"], ["yes"])))
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)
