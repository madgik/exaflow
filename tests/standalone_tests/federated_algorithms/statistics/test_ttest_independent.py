import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.weightstats import CompareMeans
from statsmodels.stats.weightstats import DescrStatsW

from exaflow.algorithms.federated.statistics.ttest_independent import (
    FederatedTTestIndependent,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

TEST_CASES = [
    {
        "name": "balanced_small_positive",
        "group_a": [1.0, 2.0, 3.0, 4.0],
        "group_b": [2.0, 2.5, 3.5, 4.5],
        "alpha": 0.05,
    },
    {
        "name": "balanced_small_negative",
        "group_a": [-3.0, -2.0, -1.0, 0.0],
        "group_b": [-2.5, -1.5, -0.5, 0.5],
        "alpha": 0.05,
    },
    {
        "name": "unequal_sizes",
        "group_a": [1.2, 1.4, 1.6, 1.8, 2.0],
        "group_b": [0.9, 1.1, 1.3],
        "alpha": 0.1,
    },
    {
        "name": "larger_effect",
        "group_a": [10.0, 11.0, 9.5, 10.5, 12.0],
        "group_b": [7.0, 6.5, 7.5, 6.0, 7.2],
        "alpha": 0.01,
    },
    {
        "name": "mixed_signs",
        "group_a": [-1.0, 0.0, 1.0, 2.0],
        "group_b": [-2.0, -1.5, 0.5, 1.5],
        "alpha": 0.05,
    },
    {
        "name": "decimals",
        "group_a": [0.12, 0.15, 0.2, 0.18, 0.16],
        "group_b": [0.05, 0.07, 0.08, 0.1, 0.09],
        "alpha": 0.05,
    },
    {
        "name": "larger_sample",
        "group_a": [
            1.0,
            1.1,
            0.9,
            1.2,
            1.3,
            1.05,
            0.95,
            1.15,
            1.25,
            1.4,
        ],
        "group_b": [
            0.8,
            0.85,
            0.9,
            0.95,
            1.0,
            0.75,
            0.88,
            0.92,
        ],
        "alpha": 0.05,
    },
    {
        "name": "integer_values",
        "group_a": [3, 4, 5, 6, 7],
        "group_b": [1, 2, 3, 4, 5],
        "alpha": 0.05,
    },
    {
        "name": "close_means",
        "group_a": [5.0, 5.1, 4.9, 5.2, 5.0],
        "group_b": [5.05, 5.0, 4.95, 5.1, 5.0],
        "alpha": 0.05,
    },
    {
        "name": "high_variance",
        "group_a": [1.0, 5.0, 9.0, 13.0, 17.0],
        "group_b": [2.0, 4.0, 6.0, 8.0, 10.0],
        "alpha": 0.05,
    },
]


class TestFederatedTTestIndependent(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        sample_a = kwargs["sample_a"]
        sample_b = kwargs["sample_b"]
        alpha = kwargs["alpha"]

        d1 = DescrStatsW(sample_a)
        d2 = DescrStatsW(sample_b)
        cm = CompareMeans(d1, d2)

        t_stat, p_value, df = cm.ttest_ind(usevar="pooled", alternative="two-sided")
        ci_lower, ci_upper = cm.tconfint_diff(
            alpha=alpha, alternative="two-sided", usevar="pooled"
        )

        mean_a = d1.mean
        mean_b = d2.mean
        mean_diff = mean_a - mean_b

        se_diff = cm.std_meandiff_pooledvar

        n_a = len(sample_a)
        n_b = len(sample_b)
        var_a = np.var(sample_a, ddof=1)
        var_b = np.var(sample_b, ddof=1)
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        cohens_d = mean_diff / np.sqrt(pooled_var)

        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "df": float(df),
            "mean_diff": float(mean_diff),
            "se_diff": float(se_diff),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "cohens_d": float(cohens_d),
        }

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        ttest = FederatedTTestIndependent(agg_client)
        return ttest.compute(
            data=X,
            group_var="group",
            value_var="value",
            group_a="A",
            group_b="B",
            alpha=kwargs["alpha"],
            alternative="two-sided",
        )

    def compare(self, federated_output, centralized_output, **kwargs):
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
            rtol=1e-9,
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
        sample_a = np.asarray(case["group_a"], dtype=float)
        sample_b = np.asarray(case["group_b"], dtype=float)
        data = pd.DataFrame(
            {
                "group": ["A"] * len(sample_a) + ["B"] * len(sample_b),
                "value": np.concatenate([sample_a, sample_b]),
            }
        )
        self.run_comparison(
            X=data,
            y=np.zeros((data.shape[0],), dtype=float),
            n_workers=1,
            alpha=case["alpha"],
            sample_a=sample_a,
            sample_b=sample_b,
        )

    @pytest.mark.parametrize(
        "case", TEST_CASES, ids=[case["name"] for case in TEST_CASES]
    )
    def test_federated_algorithm_with_multiple_workers(self, case):
        sample_a = np.asarray(case["group_a"], dtype=float)
        sample_b = np.asarray(case["group_b"], dtype=float)
        data = pd.DataFrame(
            {
                "group": ["A"] * len(sample_a) + ["B"] * len(sample_b),
                "value": np.concatenate([sample_a, sample_b]),
            }
        )
        self.run_comparison(
            X=data,
            y=np.zeros((data.shape[0],), dtype=float),
            n_workers=3,
            alpha=case["alpha"],
            sample_a=sample_a,
            sample_b=sample_b,
        )
