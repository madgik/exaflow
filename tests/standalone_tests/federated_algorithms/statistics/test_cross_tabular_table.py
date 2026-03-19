import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.statistics.cross_tabular_table import CrossTabTable
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

np.random.seed(42)

TEST_CASES = [
    {
        "name": "basic",
        "dataset": pd.DataFrame(
            {
                "factor": np.random.choice(["A", "B", "C"], size=20),
                "outcome": np.random.choice(["X", "Y", "Z"], size=20),
            }
        ),
        "factor_categories": ["A", "B", "C"],
        "outcome_categories": ["X", "Y", "Z"],
        "dropna": False,
    },
    {
        "name": "missing_categories",
        "dataset": pd.DataFrame(
            {
                "factor": np.random.choice(["A", "B"], size=20),
                "outcome": np.random.choice(["X", "Y"], size=20),
            }
        ),
        "factor_categories": ["A", "B", "C"],
        "outcome_categories": ["X", "Y", "Z"],
        "dropna": False,
    },
    {
        "name": "with_nans_dropna_false",
        "dataset": pd.DataFrame(
            {
                "factor": pd.Series(
                    ["A", "B", "C", None, "A", "B", "C", None, None, "A"]
                ),
                "outcome": pd.Series(
                    ["X", "Y", "Z", None, "X", "Y", "Z", "X", None, None]
                ),
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
                "factor": pd.Series(
                    ["A", "B", "C", None, "A", "B", "C", None, None, None]
                ),
                "outcome": pd.Series(
                    ["X", "Y", "Z", None, "X", "Y", "Z", None, None, None]
                ),
            }
        ),
        "factor_categories": ["A", "B", "C"],
        "outcome_categories": ["X", "Y", "Z"],
        "dropna": True,
    },
    {
        "name": "missing_categories_per_client",
        # Specifically engineer dataset so partition splits completely miss some categories
        "dataset": pd.DataFrame(
            {
                "factor": [4] * 8 + [5] * 8 + [7] * 8,
                "outcome": ["X", "Y"] * 4 + ["X"] * 8 + ["Z"] * 8,
            }
        ),
        "factor_categories": [4, 5, 7],
        "outcome_categories": ["X", "Y", "Z"],
        "dropna": False,
    },
    {
        "name": "missing_categories_per_client_with_nans",
        # Some values missing in specific clients, plus nulls
        "dataset": pd.DataFrame(
            {
                "factor": pd.Series(
                    [4, 4, None, None, 5, 5, None, None, 7, 7, None, None]
                ),
                "outcome": pd.Series(
                    ["X", "Y", None, None, "X", "X", None, None, "Z", "Z", None, None]
                ),
            }
        ),
        "factor_categories": [4, 5, 7],
        "outcome_categories": ["X", "Y", "Z"],
        "dropna": False,
    },
    {
        "name": "empty_dataset",
        "dataset": pd.DataFrame(
            {
                "factor": pd.Series([], dtype=str),
                "outcome": pd.Series([], dtype=str),
            }
        ),
        "factor_categories": ["A", "B"],
        "outcome_categories": ["X", "Y"],
        "dropna": False,
    },
    {
        "name": "categories_none",
        "dataset": pd.DataFrame(
            {
                "factor": np.random.choice(["A", "B", "C"], size=20),
                "outcome": np.random.choice(["X", "Y", "Z"], size=20),
            }
        ),
        "factor_categories": None,
        "outcome_categories": None,
        "dropna": False,
    },
    {
        "name": "categories_none_with_missing_per_client",
        "dataset": pd.DataFrame(
            {
                "factor": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
                "outcome": ["X", "Y"] * 4 + ["Y", "Z"] * 4 + ["Z", "W"] * 4,
            }
        ),
        "factor_categories": None,
        "outcome_categories": None,
        "dropna": False,
    },
    {
        "name": "numeric_categories_explicit",
        "dataset": pd.DataFrame(
            {
                "factor": np.random.choice([1, 2, 3, 4], size=20),
                "outcome": np.random.choice([10, 20, 30], size=20),
            }
        ),
        "factor_categories": [1, 2, 3, 4, 5],
        "outcome_categories": [10, 20, 30, 40],
        "dropna": False,
    },
    {
        "name": "numeric_categories_none",
        "dataset": pd.DataFrame(
            {
                "factor": np.random.choice([1, 2, 3], size=20),
                "outcome": np.random.choice([10, 20, 30], size=20),
            }
        ),
        "factor_categories": None,
        "outcome_categories": None,
        "dropna": False,
    },
    {
        "name": "categories_none_with_nans_dropna_false",
        "dataset": pd.DataFrame(
            {
                "factor": pd.Series(
                    ["A", "B", None, "A", None, "B", "C", None, None, "A"]
                ),
                "outcome": pd.Series(
                    ["X", "Y", "Z", None, "X", "Y", None, "X", None, None]
                ),
            }
        ),
        "factor_categories": None,
        "outcome_categories": None,
        "dropna": False,
    },
    {
        "name": "categories_none_with_nans_dropna_true",
        "dataset": pd.DataFrame(
            {
                "factor": pd.Series([1, 2, None, 1, None, 2, 3, None, None, 1]),
                "outcome": pd.Series([10, 20, 30, None, 10, 20, None, 10, None, None]),
            }
        ),
        "factor_categories": None,
        "outcome_categories": None,
        "dropna": True,
    },
]


class TestFederatedCrossTabTable(FederatedAlgorithmTest):
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
        return pd.crosstab(df[factor], df[outcome], dropna=dropna)

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
        cross_tab_algo = CrossTabTable(aggregator)

        # Assuming the federated algorithm allows passing dropna
        # If it doesn't currently, it will need to be added.
        return cross_tab_algo.compute(
            x.copy(),
            factor=factor,
            outcome=outcome,
            factor_categories=factor_categories,
            outcome_categories=outcome_categories,
            dropna=dropna if dropna is not None else False,
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        pd.testing.assert_frame_equal(federated_output, centralized_output)

    def _outputs_equal(self, left, right):
        if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
            return left.equals(right)
        return super()._outputs_equal(left, right)

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
    def test_federated_algorithm_with_one_worker(self, case):
        dataset = case["dataset"]
        self.run_comparison(
            X=dataset,
            y=dataset["outcome"],  # Dummy y just to satisfy run_comparison
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
