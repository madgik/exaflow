"""Variance-based statistics must stay accurate when the mean dwarfs the spread.

Raw-moment formulas (n * sum(x^2) - sum(x)^2) lose every significant digit for
values around 1e8 with unit standard deviation.
"""

import numpy as np
import pandas as pd
import scipy.stats as st

from exaflow.algorithms.federated.statistics.describe import FederatedDescribe
from exaflow.algorithms.federated.statistics.pearson_correlation import (
    FederatedPearsonCorrelation,
)
from exaflow.algorithms.federated.statistics.ttest_independent import (
    FederatedTTestIndependent,
)
from exaflow.algorithms.federated.statistics.ttest_onesample import (
    FederatedTTestOneSample,
)
from exaflow.algorithms.federated.statistics.ttest_paired import FederatedTTestPaired
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    _simulate_federated_execution,
)

OFFSET = 1e8
N_ROWS = 300
N_WORKERS = 3
RTOL = 1e-6

_rng = np.random.default_rng(0)
X = OFFSET + _rng.standard_normal(N_ROWS)
Y = OFFSET + 0.5 * (X - OFFSET) + _rng.standard_normal(N_ROWS)
PARTS = np.array_split(np.arange(N_ROWS), N_WORKERS)


def _run(worker_fn):
    outputs = _simulate_federated_execution(
        N_WORKERS, lambda worker_id, agg_client: worker_fn(PARTS[worker_id], agg_client)
    )
    return outputs[0]


def test_ttest_onesample_large_offset():
    result = _run(
        lambda rows, agg_client: FederatedTTestOneSample(agg_client).compute(
            sample=X[rows], mu=OFFSET, alpha=0.05, alternative="two-sided"
        )
    )
    expected = st.ttest_1samp(X, OFFSET)
    np.testing.assert_allclose(result["std"], np.std(X, ddof=1), rtol=RTOL)
    np.testing.assert_allclose(result["t_stat"], expected.statistic, rtol=RTOL)


def test_ttest_paired_large_offset():
    y = Y - OFFSET
    result = _run(
        lambda rows, agg_client: FederatedTTestPaired(agg_client).compute(
            sample_x=X[rows], sample_y=y[rows], alpha=0.05, alternative="two-sided"
        )
    )
    expected = st.ttest_rel(X, y)
    np.testing.assert_allclose(result["t_stat"], expected.statistic, rtol=RTOL)


def test_ttest_independent_large_offset():
    data = pd.DataFrame(
        {
            "value": np.concatenate([X, Y]),
            "group": ["a"] * N_ROWS + ["b"] * N_ROWS,
        }
    )
    result = _run(
        lambda rows, agg_client: FederatedTTestIndependent(agg_client).compute(
            data=data.iloc[np.concatenate([rows, rows + N_ROWS])],
            group_var="group",
            value_var="value",
            group_a="a",
            group_b="b",
            alpha=0.05,
            alternative="two-sided",
        )
    )
    pooled_var = (np.var(X, ddof=1) + np.var(Y, ddof=1)) / 2
    expected_se = np.sqrt(pooled_var * 2 / N_ROWS)
    np.testing.assert_allclose(result["se_diff"], expected_se, rtol=RTOL)


def test_pearson_large_offset():
    data = pd.DataFrame({"x": X, "y": Y})
    result = _run(
        lambda rows, agg_client: FederatedPearsonCorrelation(agg_client).corrcoef(
            data=data.iloc[rows], x_vars=["x"], y_vars=["y"], alpha=0.05
        )
    )
    np.testing.assert_allclose(
        result.correlations[0][0], np.corrcoef(X, Y)[0, 1], rtol=RTOL
    )


def test_describe_large_offset():
    # Two datasets per worker exercise the per-dataset combination as well.
    data = pd.DataFrame(
        {"value": X, "dataset": np.where(np.arange(N_ROWS) % 2, "d1", "d2")}
    )
    result = _run(
        lambda rows, agg_client: FederatedDescribe(agg_client).describe(
            data=data.iloc[rows],
            numerical_vars=["value"],
            nominal_vars=[],
            min_row_count=1,
            nominal_levels={},
        )
    )
    np.testing.assert_allclose(
        result.global_recs[0]["data"]["std"], np.std(X, ddof=1), rtol=RTOL
    )
