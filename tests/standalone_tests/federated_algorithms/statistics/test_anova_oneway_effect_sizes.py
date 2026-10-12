import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from exaflow.algorithms.federated.statistics.anova_oneway import FederatedAnovaOneWay
from exaflow.algorithms.federated.statistics.anova_oneway import _effect_sizes
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


def test_effect_sizes_match_statsmodels_anova_table():
    groups = [
        np.array([1.0, 2.0, 3.5, 2.2]),
        np.array([4.0, 5.5, 6.1, 7.0, 5.0]),
        np.array([2.0, 3.0, 2.5]),
    ]
    model = FederatedAnovaOneWay(DummyAggClient()).fit(
        groups=groups, categories=["a", "b", "c"]
    )

    frame = pd.DataFrame(
        {
            "y": np.concatenate(groups),
            "g": np.repeat(["a", "b", "c"], [len(g) for g in groups]),
        }
    )
    table = sm.stats.anova_lm(ols("y ~ C(g)", data=frame).fit(), typ=2)
    ss_between, ss_within = table["sum_sq"]
    df_between = table["df"].iloc[0]
    ms_within = ss_within / table["df"].iloc[1]
    ss_total = ss_between + ss_within

    np.testing.assert_allclose(model.eta_squared, ss_between / ss_total, rtol=1e-10)
    np.testing.assert_allclose(
        model.omega_squared,
        (ss_between - df_between * ms_within) / (ss_total + ms_within),
        rtol=1e-10,
    )


def test_effect_sizes_are_none_without_variation():
    assert _effect_sizes(
        ss_between=0.0, ss_within=0.0, df_between=1.0, ms_within=0.0
    ) == (None, None)


def test_welch_anova_matches_statsmodels():
    from statsmodels.stats.oneway import anova_oneway

    groups = [
        np.array([1.0, 2.0, 3.5, 2.2]),
        np.array([4.0, 5.5, 6.1, 7.0, 5.0]),
        np.array([2.0, 3.0, 2.5]),
    ]
    model = FederatedAnovaOneWay(DummyAggClient()).fit(
        groups=groups, categories=["a", "b", "c"]
    )
    reference = anova_oneway(groups, use_var="unequal", welch_correction=True)

    np.testing.assert_allclose(model.welch_f_stat, reference.statistic, rtol=1e-10)
    np.testing.assert_allclose(model.welch_p_value, reference.pvalue, rtol=1e-10)
    np.testing.assert_allclose(
        [model.welch_df_between, model.welch_df_within], reference.df, rtol=1e-10
    )


def test_welch_anova_is_none_for_single_observation_group():
    model = FederatedAnovaOneWay(DummyAggClient()).fit(
        groups=[np.array([1.0, 2.0, 4.0]), np.array([5.0])], categories=["a", "b"]
    )

    assert model.welch_f_stat is None
    assert model.welch_p_value is None
