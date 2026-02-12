import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.weightstats import DescrStatsW

from exaflow.algorithms.federated.statistics.ttest_onesample import (
    FederatedTTestOneSample,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
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


class TestFederatedTTestOneSample(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        sample = kwargs["sample"]
        mu = kwargs["mu"]
        alpha = kwargs["alpha"]

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

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        ttest = FederatedTTestOneSample(agg_client)
        return ttest.compute(
            sample=X["value"],
            mu=kwargs["mu"],
            alpha=kwargs["alpha"],
            alternative="two-sided",
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        np.testing.assert_allclose(
            federated_output["n_obs"],
            centralized_output["n_obs"],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            federated_output["t_stat"],
            centralized_output["t_stat"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            federated_output["p_value"],
            centralized_output["p_value"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            federated_output["df"],
            centralized_output["df"],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            federated_output["mean_diff"],
            centralized_output["mean_diff"],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            federated_output["se_diff"],
            centralized_output["se_diff"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            federated_output["std"],
            centralized_output["std"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            federated_output["ci_lower"],
            centralized_output["ci_lower"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            federated_output["ci_upper"],
            centralized_output["ci_upper"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            federated_output["cohens_d"],
            centralized_output["cohens_d"],
            rtol=1e-7,
            atol=1e-10,
        )

    @pytest.mark.parametrize(
        "case", TEST_CASES, ids=[case["name"] for case in TEST_CASES]
    )
    def test_federated_algorithm_with_one_worker(self, case):
        sample = np.asarray(case["values"], dtype=float)
        df = pd.DataFrame({"value": sample})
        self.run_comparison(
            X=df,
            y=np.zeros((df.shape[0],), dtype=float),
            n_workers=1,
            sample=sample,
            mu=case["mu"],
            alpha=case["alpha"],
        )

    @pytest.mark.parametrize(
        "case", TEST_CASES, ids=[case["name"] for case in TEST_CASES]
    )
    def test_federated_algorithm_with_multiple_workers(self, case):
        sample = np.asarray(case["values"], dtype=float)
        df = pd.DataFrame({"value": sample})
        self.run_comparison(
            X=df,
            y=np.zeros((df.shape[0],), dtype=float),
            n_workers=3,
            sample=sample,
            mu=case["mu"],
            alpha=case["alpha"],
        )
