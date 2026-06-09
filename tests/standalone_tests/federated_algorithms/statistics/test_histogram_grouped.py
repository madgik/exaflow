import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.sql.sql import FederatedSQL
from exaflow.algorithms.federated.statistics.histogram import SimpleHistogram
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

HIST_DATASETS = [
    {
        "name": "complex_10_elements_3_groups",
        "df": pd.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "val": [
                    1.0,
                    1.5,
                    2.0,
                    2.5,  # Group A
                    10.0,
                    15.0,
                    20.0,  # Group B
                    100.0,
                    200.0,
                    300.0,  # Group C
                ],
            }
        ),
        "num_bins": 3,
    }
]


class TestFederatedSQLHistogram(FederatedAlgorithmTest):
    """Test suite for Histogram with Group By, following the TestFederatedSQL pattern."""

    def _split_inputs(self, X, y, n_workers: int):
        """Split dataframe into n_workers parts."""
        parts = np.array_split(np.arange(len(X)), n_workers)
        x_parts = [X.iloc[idx].reset_index(drop=True) for idx in parts]
        y_array = np.asarray(y)
        y_parts = [y_array[idx] for idx in parts]
        return x_parts, y_parts, X, y_array

    def compute_centralized_result(self, X, y, **kwargs):
        """Compute ground truth histograms per group."""
        num_bins = kwargs["num_bins"]
        results = {}
        for group_key, sub in X.groupby("group", sort=True):
            vals = sub["val"].to_numpy(dtype=float)
            min_val = np.nanmin(vals)
            max_val = np.nanmax(vals)
            if min_val == max_val:
                max_val = min_val + 1.0
            counts, edges = np.histogram(vals, bins=num_bins, range=(min_val, max_val))
            results[group_key] = (counts.astype(float), edges)
        return results

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        """Compute federated histograms per group using FederatedSQL."""
        num_bins = kwargs["num_bins"]
        aggregator = NumpyAggregator(agg_client)
        hist_algo = SimpleHistogram(aggregator)

        def hist_func(x):
            return hist_algo.compute(x, num_bins=num_bins)

        results = (
            FederatedSQL(X, aggregator)
            .group_by("group")
            .aggregate("hist", hist_func, "val")
            .run()
        )
        # Convert dataframe to dict {group: (counts, edges)}
        return results.dataframe["hist"].to_dict()

    def compare(self, federated_output, centralized_output, **kwargs):
        """Compare dict of histograms."""
        assert set(federated_output.keys()) == set(centralized_output.keys())
        for key in centralized_output:
            fed_counts, fed_edges = federated_output[key]
            cnt_counts, cnt_edges = centralized_output[key]

            assert np.array_equal(fed_counts, cnt_counts)
            assert np.allclose(fed_edges, cnt_edges, atol=1e-7)

    @pytest.mark.parametrize(
        "dataset", HIST_DATASETS, ids=[d["name"] for d in HIST_DATASETS]
    )
    def test_federated_algorithm_with_one_worker(self, dataset):
        df = dataset["df"]
        self.run_comparison(
            X=df,
            y=np.zeros(len(df), dtype=float),
            n_workers=1,
            num_bins=dataset["num_bins"],
        )

    @pytest.mark.parametrize(
        "dataset", HIST_DATASETS, ids=[d["name"] for d in HIST_DATASETS]
    )
    def test_federated_algorithm_with_multiple_workers(self, dataset):
        df = dataset["df"]
        self.run_comparison(
            X=df,
            y=np.zeros(len(df), dtype=float),
            n_workers=3,
            num_bins=dataset["num_bins"],
        )
