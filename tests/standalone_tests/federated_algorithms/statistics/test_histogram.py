import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.statistics.histogram import FederatedHistogram
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)

NUMERICAL_CASES = [
    {
        "y": [0.1, 0.2, 0.3, 0.5, 1.0, 1.2, 1.4, 1.8, 2.0, 2.2],
        "group": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "bins": 5,
    },
    {
        "y": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        "group": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
        "bins": 4,
    },
    {
        "y": [10, 11, 12, 13, 14, 15, 16, 17],
        "group": ["A", "A", "A", "B", "B", "C", "C", "C"],
        "bins": 6,
    },
    {
        "y": [0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
        "group": ["A", "B", "A", "B", "A", "B", "C", "C"],
        "bins": 3,
    },
    {
        "y": [5.5, 5.7, 5.9, 6.1, 6.3, 6.5, 6.7, 6.9, 7.1, 7.3],
        "group": ["C", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "bins": 8,
    },
    {
        "y": [100, 100, 101, 102, 103, 103, 104, 105, 106, 107],
        "group": ["A", "A", "A", "B", "B", "C", "C", "C", "B", "A"],
        "bins": 5,
    },
    {
        "y": [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5],
        "group": ["B", "B", "B", "C", "C", "A", "A", "A"],
        "bins": 4,
    },
    {
        "y": [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0],
        "group": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
        "bins": 7,
    },
    {
        "y": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        "group": ["A", "B", "C", "A", "B", "C"],
        "bins": 3,
    },
    {
        "y": [9.9, 9.8, 9.7, 9.6, 9.5, 9.4, 9.3, 9.2],
        "group": ["C", "B", "A", "C", "B", "A", "C", "B"],
        "bins": 4,
    },
]


class TestFederatedHistogramNumerical(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        df = X
        bins = kwargs["bins"]
        min_row_count = kwargs["min_row_count"]

        y_values = df["y"].to_numpy(dtype=float)
        global_min = float(np.min(y_values))
        global_max = float(np.max(y_values))
        if global_min == global_max:
            global_max = global_min + 1.0
        bin_edges = np.linspace(global_min, global_max, bins + 1)
        expected_counts, _ = np.histogram(y_values, bins=bin_edges)
        expected_counts = [
            count if count >= min_row_count else None
            for count in expected_counts.tolist()
        ]

        grouped_expected = []
        for group in ["A", "B", "C"]:
            subset = df.loc[df["group"] == group, "y"].to_numpy(dtype=float)
            expected_group_counts, _ = np.histogram(subset, bins=bin_edges)
            expected_group_counts = [
                count if count >= min_row_count else None
                for count in expected_group_counts.tolist()
            ]
            grouped_expected.append(expected_group_counts)

        return {
            "bin_edges": bin_edges.tolist(),
            "counts": expected_counts,
            "grouped_counts": grouped_expected,
        }

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        hist = FederatedHistogram(agg_client=agg_client)
        return hist.hist(
            data=X,
            y_var="y",
            x_vars=["group"],
            metadata=kwargs["metadata"],
            bins=kwargs["bins"],
            min_row_count=kwargs["min_row_count"],
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        assert federated_output.bins == pytest.approx(centralized_output["bin_edges"])
        assert federated_output.counts == centralized_output["counts"]

        grouped = federated_output.grouped["group"]
        assert grouped.groups == ["A", "B", "C"]
        for counts, expected in zip(
            grouped.counts, centralized_output["grouped_counts"]
        ):
            assert counts == expected

    @pytest.mark.parametrize("case", NUMERICAL_CASES)
    def test_federated_algorithm_with_one_worker(self, case):
        df = pd.DataFrame({"y": case["y"], "group": case["group"]})
        metadata = {
            "y": {"is_categorical": False},
            "group": {
                "is_categorical": True,
                "enumerations": {"A": "A", "B": "B", "C": "C"},
            },
        }
        self.run_comparison(
            X=df,
            y=np.zeros((df.shape[0],), dtype=float),
            n_workers=1,
            bins=case["bins"],
            min_row_count=1,
            metadata=metadata,
        )

    @pytest.mark.parametrize("case", NUMERICAL_CASES)
    def test_federated_algorithm_with_multiple_workers(self, case):
        df = pd.DataFrame({"y": case["y"], "group": case["group"]})
        metadata = {
            "y": {"is_categorical": False},
            "group": {
                "is_categorical": True,
                "enumerations": {"A": "A", "B": "B", "C": "C"},
            },
        }
        self.run_comparison(
            X=df,
            y=np.zeros((df.shape[0],), dtype=float),
            n_workers=3,
            bins=case["bins"],
            min_row_count=1,
            metadata=metadata,
        )


CATEGORICAL_CASES = [
    {
        "y": ["low", "mid", "high", "low", "mid", "high"],
        "group": ["A", "A", "A", "B", "B", "C"],
    },
    {
        "y": ["low", "low", "low", "mid", "mid", "high"],
        "group": ["A", "B", "C", "A", "B", "C"],
    },
    {
        "y": ["mid", "mid", "high", "high", "high", "low"],
        "group": ["C", "B", "A", "A", "B", "C"],
    },
    {
        "y": ["low", "mid", "mid", "mid", "high", "high"],
        "group": ["A", "A", "B", "B", "C", "C"],
    },
    {
        "y": ["high", "high", "high", "high", "mid", "low"],
        "group": ["A", "B", "C", "A", "B", "C"],
    },
    {
        "y": ["low", "mid", "high", "low", "mid", "high", "low"],
        "group": ["A", "A", "A", "B", "B", "C", "C"],
    },
    {
        "y": ["mid", "mid", "mid", "mid", "mid", "mid"],
        "group": ["A", "B", "C", "A", "B", "C"],
    },
    {
        "y": ["low", "high", "low", "high", "low", "high"],
        "group": ["A", "B", "C", "A", "B", "C"],
    },
    {
        "y": ["low", "mid", "high", "low", "mid", "high", "mid", "mid"],
        "group": ["A", "A", "B", "B", "C", "C", "A", "B"],
    },
    {
        "y": ["high", "mid", "low", "high", "mid", "low", "high"],
        "group": ["C", "C", "C", "B", "B", "A", "A"],
    },
]


class TestFederatedHistogramCategorical(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        df = X
        y_levels = kwargs["y_levels"]
        min_row_count = kwargs["min_row_count"]

        expected_counts = [df["y"].value_counts().get(level, 0) for level in y_levels]
        expected_counts = [
            count if count >= min_row_count else None for count in expected_counts
        ]

        grouped_expected = []
        for group in ["A", "B", "C"]:
            subset = df.loc[df["group"] == group, "y"]
            expected_group_counts = [
                subset.value_counts().get(level, 0) for level in y_levels
            ]
            expected_group_counts = [
                count if count >= min_row_count else None
                for count in expected_group_counts
            ]
            grouped_expected.append(expected_group_counts)

        return {
            "bins": y_levels,
            "counts": expected_counts,
            "grouped_counts": grouped_expected,
        }

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        hist = FederatedHistogram(agg_client=agg_client)
        return hist.hist(
            data=X,
            y_var="y",
            x_vars=["group"],
            metadata=kwargs["metadata"],
            bins=10,
            min_row_count=kwargs["min_row_count"],
        )

    def compare(self, federated_output, centralized_output, **kwargs):
        assert federated_output.bins == centralized_output["bins"]
        assert federated_output.counts == centralized_output["counts"]

        grouped = federated_output.grouped["group"]
        assert grouped.groups == ["A", "B", "C"]
        for counts, expected in zip(
            grouped.counts, centralized_output["grouped_counts"]
        ):
            assert counts == expected

    @pytest.mark.parametrize("case", CATEGORICAL_CASES)
    def test_federated_algorithm_with_one_worker(self, case):
        y_levels = ["low", "mid", "high"]
        df = pd.DataFrame({"y": case["y"], "group": case["group"]})
        metadata = {
            "y": {
                "is_categorical": True,
                "enumerations": {level: level for level in y_levels},
            },
            "group": {
                "is_categorical": True,
                "enumerations": {"A": "A", "B": "B", "C": "C"},
            },
        }
        self.run_comparison(
            X=df,
            y=np.zeros((df.shape[0],), dtype=float),
            n_workers=1,
            metadata=metadata,
            y_levels=y_levels,
            min_row_count=1,
        )

    @pytest.mark.parametrize("case", CATEGORICAL_CASES)
    def test_federated_algorithm_with_multiple_workers(self, case):
        y_levels = ["low", "mid", "high"]
        df = pd.DataFrame({"y": case["y"], "group": case["group"]})
        metadata = {
            "y": {
                "is_categorical": True,
                "enumerations": {level: level for level in y_levels},
            },
            "group": {
                "is_categorical": True,
                "enumerations": {"A": "A", "B": "B", "C": "C"},
            },
        }
        self.run_comparison(
            X=df,
            y=np.zeros((df.shape[0],), dtype=float),
            n_workers=3,
            metadata=metadata,
            y_levels=y_levels,
            min_row_count=1,
        )
