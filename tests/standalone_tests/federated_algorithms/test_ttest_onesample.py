import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.weightstats import DescrStatsW

from exaflow.algorithms.federated.ttest_onesample import FederatedTTestOneSample
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)

TEST_CASES = [
    {
        "name": "small_positive",
        "values": [1.0, 2.0, 3.0, 4.0],
        "mu": 0.0,
        "alpha": 0.05,
    },
    {
        "name": "small_negative",
        "values": [-3.0, -2.0, -1.0, 0.0],
        "mu": -1.5,
        "alpha": 0.05,
    },
    {
        "name": "unequal_mu",
        "values": [1.2, 1.4, 1.6, 1.8, 2.0],
        "mu": 1.0,
        "alpha": 0.1,
    },
    {
        "name": "larger_effect",
        "values": [10.0, 11.0, 9.5, 10.5, 12.0],
        "mu": 8.0,
        "alpha": 0.01,
    },
    {
        "name": "mixed_signs",
        "values": [-1.0, 0.0, 1.0, 2.0],
        "mu": 0.5,
        "alpha": 0.05,
    },
    {
        "name": "decimals",
        "values": [0.12, 0.15, 0.2, 0.18, 0.16],
        "mu": 0.1,
        "alpha": 0.05,
    },
    {
        "name": "larger_sample",
        "values": [1.0, 1.1, 0.9, 1.2, 1.3, 1.05, 0.95, 1.15, 1.25, 1.4],
        "mu": 1.0,
        "alpha": 0.05,
    },
    {
        "name": "integer_values",
        "values": [3, 4, 5, 6, 7],
        "mu": 5.0,
        "alpha": 0.05,
    },
    {
        "name": "close_mean",
        "values": [5.0, 5.1, 4.9, 5.2, 5.0],
        "mu": 5.0,
        "alpha": 0.05,
    },
    {
        "name": "high_variance",
        "values": [1.0, 5.0, 9.0, 13.0, 17.0],
        "mu": 7.0,
        "alpha": 0.05,
    },
]


def _statsmodels_expected(sample, *, mu: float, alpha: float):
    stats = DescrStatsW(sample)
    t_stat, p_value, df = stats.ttest_mean(value=mu, alternative="two-sided")
    ci_lower, ci_upper = stats.tconfint_mean(alpha=alpha, alternative="two-sided")

    mean_val = float(stats.mean)
    sd = float(np.std(sample, ddof=1))
    se = sd / np.sqrt(len(sample))
    cohens_d = (mean_val - mu) / sd

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "df": float(df),
        "mean_diff": mean_val,
        "se_diff": se,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "cohens_d": float(cohens_d),
        "std": sd,
        "n_obs": int(stats.nobs),
    }


@pytest.mark.parametrize("case", TEST_CASES, ids=[case["name"] for case in TEST_CASES])
def test_federated_one_sample_ttest_matches_statsmodels(case):
    sample = np.asarray(case["values"], dtype=float)
    mu = case["mu"]
    alpha = case["alpha"]

    df = pd.DataFrame({"value": sample})

    agg_client = DummyAggClient()
    ttest = FederatedTTestOneSample(agg_client)
    result = ttest.compute(
        sample=df["value"],
        mu=mu,
        alpha=alpha,
        alternative="two-sided",
    )

    expected = _statsmodels_expected(sample, mu=mu, alpha=alpha)

    np.testing.assert_allclose(
        result["n_obs"], expected["n_obs"], rtol=1e-12, atol=1e-12
    )
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
    np.testing.assert_allclose(result["std"], expected["std"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(
        result["ci_lower"], expected["ci_lower"], rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        result["ci_upper"], expected["ci_upper"], rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        result["cohens_d"], expected["cohens_d"], rtol=1e-7, atol=1e-10
    )
