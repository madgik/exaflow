import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from exaflow.algorithms.federated.statistics.ttest_welch import FederatedTtestWelch
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

TEST_CASES = [
    {
        "name": "balanced_unequal_variance",
        "group_a": [1.0, 1.2, 0.8, 1.1, 0.9],
        "group_b": [0.2, 1.8, 1.4, 0.4, 1.6],
        "alpha": 0.05,
        "alternative": "two-sided",
    },
    {
        "name": "unequal_sizes",
        "group_a": [10.1, 10.3, 9.9, 10.0, 10.4, 9.8],
        "group_b": [8.2, 9.5, 7.9],
        "alpha": 0.1,
        "alternative": "greater",
    },
    {
        "name": "left_tail",
        "group_a": [-2.1, -1.7, -1.5, -1.9],
        "group_b": [0.2, -0.3, 0.1, 0.4, -0.1],
        "alpha": 0.05,
        "alternative": "less",
    },
    {
        "name": "mixed_signs",
        "group_a": [-1.0, 0.0, 1.0, 2.0, 3.0],
        "group_b": [-2.0, -1.5, 0.5, 1.5, 4.0, 4.5],
        "alpha": 0.05,
        "alternative": "two-sided",
    },
]


class TestFederatedTtestWelch(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        return _centralized_welch(
            sample_a=kwargs["sample_a"],
            sample_b=kwargs["sample_b"],
            alpha=kwargs["alpha"],
            alternative=kwargs["alternative"],
        )

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        ttest = FederatedTtestWelch(agg_client)
        return ttest.compute(
            data=X,
            group_var="group",
            value_var="value",
            group_a="A",
            group_b="B",
            alpha=kwargs["alpha"],
            alternative=kwargs["alternative"],
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        for key in ["t_stat", "p_value", "df", "mean_diff", "se_diff", "cohens_d"]:
            np.testing.assert_allclose(
                federated_output[key],
                centralized_output[key],
                rtol=1e-9,
                atol=1e-12,
            )

        _assert_interval_close(
            federated_output["ci_lower"], centralized_output["ci_lower"]
        )
        _assert_interval_close(
            federated_output["ci_upper"], centralized_output["ci_upper"]
        )

    @pytest.mark.parametrize(
        "case", TEST_CASES, ids=[case["name"] for case in TEST_CASES]
    )
    def test_federated_algorithm_with_one_worker(self, case):
        self._run_case(case, n_workers=1)

    @pytest.mark.parametrize(
        "case", TEST_CASES, ids=[case["name"] for case in TEST_CASES]
    )
    def test_federated_algorithm_with_multiple_workers(self, case):
        self._run_case(case, n_workers=3)

    def _run_case(self, case, *, n_workers):
        sample_a = np.asarray(case["group_a"], dtype=float)
        sample_b = np.asarray(case["group_b"], dtype=float)
        data = pd.DataFrame(
            {
                "group": ["A"] * sample_a.size + ["B"] * sample_b.size,
                "value": np.concatenate([sample_a, sample_b]),
            }
        )
        self.run_comparison(
            X=data,
            y=np.zeros((data.shape[0],), dtype=float),
            n_workers=n_workers,
            sample_a=sample_a,
            sample_b=sample_b,
            alpha=case["alpha"],
            alternative=case["alternative"],
        )


def _centralized_welch(*, sample_a, sample_b, alpha, alternative):
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)

    t_stat, p_value = st.ttest_ind(
        sample_a,
        sample_b,
        equal_var=False,
        alternative=alternative,
    )
    mean_a = float(np.mean(sample_a))
    mean_b = float(np.mean(sample_b))
    var_a = float(np.var(sample_a, ddof=1))
    var_b = float(np.var(sample_b, ddof=1))
    n_a = float(sample_a.size)
    n_b = float(sample_b.size)

    mean_diff = mean_a - mean_b
    var_term_a = var_a / n_a
    var_term_b = var_b / n_b
    se_diff = float(np.sqrt(var_term_a + var_term_b))
    df = ((var_term_a + var_term_b) ** 2) / (
        (var_term_a**2 / (n_a - 1.0)) + (var_term_b**2 / (n_b - 1.0))
    )
    ci_lower, ci_upper = st.t.interval(1.0 - alpha, df, loc=mean_diff, scale=se_diff)
    if alternative == "greater":
        ci_upper = "Infinity"
    elif alternative == "less":
        ci_lower = "-Infinity"

    cohens_d = mean_diff / np.sqrt((var_a + var_b) / 2.0)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "df": float(df),
        "mean_diff": float(mean_diff),
        "se_diff": float(se_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": float(cohens_d),
    }


def _assert_interval_close(left, right):
    if isinstance(left, str) or isinstance(right, str):
        assert left == right
        return
    np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-12)
