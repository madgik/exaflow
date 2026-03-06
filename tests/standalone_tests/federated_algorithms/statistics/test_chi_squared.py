import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from exaflow.algorithms.federated.statistics.chi_squared import ChiSquared
from exaflow.algorithms.federated.utils.aggregators import NumpyAggregator
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

np.random.seed(42)

# Reuse TEST_CASES from test_cross_tab_table.py or define similar ones
# Note: chi2_contingency requires at least 2 non-zero entries in each dimension
# Some edge cases like empty_dataset or all-zero rows might fail chi2_contingency
TEST_CASES = [
    {
        "name": "basic",
        "dataset": pd.DataFrame(
            {
                "factor": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
                "outcome": ["X", "Y", "X", "Y", "X", "Y", "X", "X", "Y", "Y"],
            }
        ),
        "factor_categories": ["A", "B", "C"],
        "outcome_categories": ["X", "Y"],
        "dropna": False,
    },
    {
        "name": "with_nans_dropna_false",
        "dataset": pd.DataFrame(
            {
                "factor": pd.Series(["A", "B", "C", None, "A", "B", "C", None, None, "A"]),
                "outcome": pd.Series(["X", "Y", "Z", None, "X", "Y", "Z", "X", None, None]),
            }
        ),
        "factor_categories": ["A", "B", "C"],
        "outcome_categories": ["X", "Y", "Z"],
        "dropna": False,
    },
     {
        "name": "with_nans_dropna_true",
        "dataset": pd.DataFrame(
            {
                "factor": pd.Series(["A", "B", "C", None, "A", "B", "C", None, None, None]),
                "outcome": pd.Series(["X", "Y", "Z", None, "X", "Y", "Z", None, None, None]),
            }
        ),
        "factor_categories": ["A", "B", "C"],
        "outcome_categories": ["X", "Y", "Z"],
        "dropna": True,
    },
]

class TestFederatedChiSquared(FederatedAlgorithmTest):
    def compute_centralized_result(self, x, y, *, factor=None, factor_categories=None, outcome=None, outcome_categories=None, dropna=None, **kwargs):
        df = x.copy()
        cross_tab = pd.crosstab(df[factor], df[outcome], dropna=dropna)
        
        # scipy.stats.chi2_contingency
        chi2, p, dof, expected = chi2_contingency(cross_tab)
        return chi2, p, dof, expected

    def compute_federated_result(self, x, y, *, agg_client, factor=None, factor_categories=None, outcome=None, outcome_categories=None, dropna=None, **kwargs):
        aggregator = NumpyAggregator(agg_client)
        chi_algo = ChiSquared(aggregator)

        return chi_algo.compute(
            x.copy(),
            factor=factor,
            factor_categories=factor_categories,
            outcome=outcome,
            outcome_categories=outcome_categories,
            dropna=dropna if dropna is not None else False,
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        f_chi2, f_p, f_dof, f_expected = federated_output
        c_chi2, c_p, c_dof, c_expected = centralized_output
        
        assert np.isclose(f_chi2, c_chi2)
        assert np.isclose(f_p, c_p)
        assert f_dof == c_dof
        np.testing.assert_allclose(f_expected, c_expected)

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
