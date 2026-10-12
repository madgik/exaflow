import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from exaflow.algorithms.federated.statistics.ttest_independent import (
    FederatedTTestIndependent,
)
from exaflow.algorithms.federated.statistics.ttest_onesample import (
    FederatedTTestOneSample,
)
from exaflow.algorithms.federated.statistics.ttest_paired import FederatedTTestPaired
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)

X = np.array([1.2, 2.5, 3.1, 4.8, 5.0, 6.3, 2.2])
Y = np.array([0.9, 2.0, 3.5, 4.0, 4.1, 5.9, 2.6])
ALPHA = 0.1


def _assert_matches_scipy(result, reference, alternative):
    ci = reference.confidence_interval(1 - ALPHA)
    np.testing.assert_allclose(result["p_value"], reference.pvalue, rtol=1e-10)
    if alternative == "greater":
        np.testing.assert_allclose(result["ci_lower"], ci.low, rtol=1e-10)
        assert result["ci_upper"] == "Infinity"
    elif alternative == "less":
        assert result["ci_lower"] == "-Infinity"
        np.testing.assert_allclose(result["ci_upper"], ci.high, rtol=1e-10)
    else:
        np.testing.assert_allclose(result["ci_lower"], ci.low, rtol=1e-10)
        np.testing.assert_allclose(result["ci_upper"], ci.high, rtol=1e-10)


ALTERNATIVES = ["two-sided", "greater", "less"]


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_independent_ttest_matches_scipy(alternative):
    data = pd.DataFrame(
        {"group": ["A"] * len(X) + ["B"] * len(Y), "value": np.concatenate([X, Y])}
    )
    result = FederatedTTestIndependent(DummyAggClient()).compute(
        data=data,
        group_var="group",
        value_var="value",
        group_a="A",
        group_b="B",
        alpha=ALPHA,
        alternative=alternative,
    )

    _assert_matches_scipy(
        result, st.ttest_ind(X, Y, alternative=alternative), alternative
    )


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_onesample_ttest_matches_scipy(alternative):
    mu = 2.0
    result = FederatedTTestOneSample(DummyAggClient()).compute(
        sample=X, mu=mu, alpha=ALPHA, alternative=alternative
    )
    reference = st.ttest_1samp(X, mu, alternative=alternative)

    # scipy reports the interval for the mean, as does the federated result.
    _assert_matches_scipy(result, reference, alternative)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_paired_ttest_matches_scipy(alternative):
    result = FederatedTTestPaired(DummyAggClient()).compute(
        sample_x=X, sample_y=Y, alpha=ALPHA, alternative=alternative
    )

    _assert_matches_scipy(
        result, st.ttest_rel(X, Y, alternative=alternative), alternative
    )


def test_independent_ttest_handles_group_of_size_one():
    data = pd.DataFrame({"group": ["A", "B", "B", "B"], "value": [1.0, 2.0, 3.0, 5.0]})
    result = FederatedTTestIndependent(DummyAggClient()).compute(
        data=data,
        group_var="group",
        value_var="value",
        group_a="A",
        group_b="B",
        alpha=0.05,
        alternative="two-sided",
    )
    reference = st.ttest_ind([1.0], [2.0, 3.0, 5.0])

    np.testing.assert_allclose(result["t_stat"], reference.statistic, rtol=1e-10)
    np.testing.assert_allclose(result["p_value"], reference.pvalue, rtol=1e-10)


def test_p_value_does_not_underflow_for_large_t_statistics():
    sample = np.linspace(100.0, 100.1, 50)
    result = FederatedTTestOneSample(DummyAggClient()).compute(
        sample=sample, mu=0.0, alpha=0.05, alternative="greater"
    )

    assert result["p_value"] > 0.0
