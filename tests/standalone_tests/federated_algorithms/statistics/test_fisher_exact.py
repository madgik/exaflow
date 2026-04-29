import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from exaflow.algorithms.federated.statistics.fisher_exact import FisherExact
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

np.random.seed(42)

# Fisher's exact test is primarily for 2x2 contingency tables
TEST_CASES = [
    {
        "name": "basic_2x2",
        "dataset": pd.DataFrame(
            {
                "factor": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
                "outcome": ["X", "Y", "X", "Y", "X", "X", "Y", "Y", "X", "Y"],
            }
        ),
        "factor_categories": ["A", "B"],
        "outcome_categories": ["X", "Y"],
        "dropna": False,
    },
    {
        "name": "with_nans_dropna_true_2x2",
        "dataset": pd.DataFrame(
            {
                "factor": pd.Series(
                    ["A", "B", None, "A", "B", None, "A", "B", None, "A"]
                ),
                "outcome": pd.Series(
                    ["X", "Y", None, "X", "Y", "X", "Y", "X", "Y", "Y"]
                ),
            }
        ),
        "factor_categories": ["A", "B"],
        "outcome_categories": ["X", "Y"],
        "dropna": True,
    },
    {
        "name": "wrong_order_labels_2x2",
        "dataset": pd.DataFrame(
            {
                "factor": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
                "outcome": ["X", "Y", "X", "Y", "X", "X", "Y", "Y", "X", "Y"],
            }
        ),
        "factor_categories": ["B", "A"],
        "outcome_categories": ["Y", "X"],
        "dropna": False,
    },
]


class TestFederatedFisherExact(FederatedAlgorithmTest):
    def compute_centralized_result(
        self,
        x,
        y,
        *,
        factor=None,
        factor_categories=None,
        outcome=None,
        outcome_categories=None,
        dropna=None,
        **kwargs,
    ):
        df = x.copy()
        cross_tab = pd.crosstab(df[factor], df[outcome], dropna=dropna)

        factor_idx = cross_tab.index
        if factor_categories is not None:
            factor_idx = pd.Index(factor_categories)
            if not dropna and df[factor].isna().any():
                factor_idx = pd.Index(list(factor_idx) + [np.nan])

        outcome_idx = cross_tab.columns
        if outcome_categories is not None:
            outcome_idx = pd.Index(outcome_categories)
            if not dropna and df[outcome].isna().any():
                outcome_idx = pd.Index(list(outcome_idx) + [np.nan])

        cross_tab = cross_tab.reindex(
            index=factor_idx, columns=outcome_idx, fill_value=0
        )
        cross_tab = cross_tab.loc[
            (cross_tab != 0).any(axis=1), (cross_tab != 0).any(axis=0)
        ]

        # scipy.stats.fisher_exact
        odds_ratio, p_value = fisher_exact(cross_tab)
        return odds_ratio, p_value

    def compute_federated_result(
        self,
        x,
        y,
        *,
        agg_client,
        factor=None,
        factor_categories=None,
        outcome=None,
        outcome_categories=None,
        dropna=None,
        **kwargs,
    ):
        aggregator = NumpyAggregator(agg_client)
        fisher_algo = FisherExact(aggregator)

        return fisher_algo.compute(
            x.copy(),
            factor=factor,
            outcome=outcome,
            factor_categories=factor_categories,
            outcome_categories=outcome_categories,
            dropna=dropna if dropna is not None else False,
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        f_odds, f_p = federated_output
        c_odds, c_p = centralized_output

        assert np.isclose(f_odds, c_odds)
        assert np.isclose(f_p, c_p)

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_one_worker(self, case):
        dataset = case["dataset"]
        self.run_comparison(
            X=dataset,
            y=dataset["outcome"],
            n_workers=1,
            factor="factor",
            factor_categories=case["factor_categories"],
            outcome="outcome",
            outcome_categories=case["outcome_categories"],
            dropna=case["dropna"],
        )

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_multiple_workers(self, case):
        dataset = case["dataset"]
        self.run_comparison(
            X=dataset,
            y=dataset["outcome"],
            n_workers=3,
            factor="factor",
            factor_categories=case["factor_categories"],
            outcome="outcome",
            outcome_categories=case["outcome_categories"],
            dropna=case["dropna"],
        )
