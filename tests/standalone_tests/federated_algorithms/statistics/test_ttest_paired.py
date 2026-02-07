import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from exaflow.algorithms.federated.statistics.ttest_paired import FederatedTTestPaired
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)

TEST_CASES = [
    {
        "name": "small_positive",
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [0.5, 1.4, 2.6, 3.7],
        "alpha": 0.05,
    },
    {
        "name": "small_negative",
        "x": [-3.0, -2.0, -1.0, 0.0],
        "y": [-2.0, -1.5, -0.5, 0.5],
        "alpha": 0.05,
    },
    {
        "name": "unequal_changes",
        "x": [1.2, 1.4, 1.6, 1.8, 2.0],
        "y": [1.1, 1.2, 1.6, 1.7, 1.9],
        "alpha": 0.1,
    },
    {
        "name": "larger_effect",
        "x": [10.0, 11.0, 9.5, 10.5, 12.0],
        "y": [7.0, 6.5, 7.5, 6.0, 7.2],
        "alpha": 0.01,
    },
    {
        "name": "mixed_signs",
        "x": [-1.0, 0.0, 1.0, 2.0],
        "y": [-1.5, 0.5, 0.5, 1.0],
        "alpha": 0.05,
    },
    {
        "name": "decimals",
        "x": [0.12, 0.15, 0.2, 0.18, 0.16],
        "y": [0.1, 0.13, 0.18, 0.17, 0.14],
        "alpha": 0.05,
    },
    {
        "name": "larger_sample",
        "x": [1.0, 1.1, 0.9, 1.2, 1.3, 1.05, 0.95, 1.15, 1.25, 1.4],
        "y": [0.9, 1.0, 0.85, 1.1, 1.2, 1.0, 0.9, 1.1, 1.2, 1.3],
        "alpha": 0.05,
    },
    {
        "name": "integer_values",
        "x": [3, 4, 5, 6, 7],
        "y": [2, 3, 5, 5, 6],
        "alpha": 0.05,
    },
    {
        "name": "close_mean",
        "x": [5.0, 5.1, 4.9, 5.2, 5.0],
        "y": [5.05, 5.0, 4.95, 5.1, 4.98],
        "alpha": 0.05,
    },
    {
        "name": "high_variance",
        "x": [1.0, 5.0, 9.0, 13.0, 17.0],
        "y": [2.0, 4.0, 8.0, 12.0, 16.0],
        "alpha": 0.05,
    },
]


def _statsmodels_expected(sample_x, sample_y, *, alpha: float):
    t_stat, p_value = st.ttest_rel(sample_x, sample_y, alternative="two-sided")
    df = len(sample_x) - 1
    diff = sample_x - sample_y
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se_diff = sd_diff / np.sqrt(len(diff))
    ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=mean_diff, scale=se_diff)
    cohens_d = mean_diff / sd_diff

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "df": float(df),
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "cohens_d": cohens_d,
    }


@pytest.mark.parametrize("case", TEST_CASES, ids=[case["name"] for case in TEST_CASES])
def test_federated_paired_ttest_matches_statsmodels(case):
    sample_x = np.asarray(case["x"], dtype=float)
    sample_y = np.asarray(case["y"], dtype=float)
    alpha = case["alpha"]

    df = pd.DataFrame({"x": sample_x, "y": sample_y})

    agg_client = DummyAggClient()
    ttest = FederatedTTestPaired(agg_client)
    result = ttest.compute(
        sample_x=df["x"],
        sample_y=df["y"],
        alpha=alpha,
        alternative="two-sided",
    )

    expected = _statsmodels_expected(sample_x, sample_y, alpha=alpha)

    np.testing.assert_allclose(
        result["t_stat"], expected["t_stat"], rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        result["p_value"], expected["p_value"], rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(result["df"], expected["df"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result["mean_diff"], expected["mean_diff"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result["se_diff"], expected["se_diff"], rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        result["ci_lower"], expected["ci_lower"], rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        result["ci_upper"], expected["ci_upper"], rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        result["cohens_d"], expected["cohens_d"], rtol=1e-7, atol=1e-10
    )
