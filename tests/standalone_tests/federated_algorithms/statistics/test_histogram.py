import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.statistics.histogram import FederatedHistogram
from tests.standalone_tests.federated_algorithms.utils import DummyAggClient

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


@pytest.mark.parametrize("case", NUMERICAL_CASES)
def test_histogram_matches_numpy(case):
    df = pd.DataFrame({"y": case["y"], "group": case["group"]})
    metadata = {
        "y": {"is_categorical": False},
        "group": {
            "is_categorical": True,
            "enumerations": {"A": "A", "B": "B", "C": "C"},
        },
    }

    bins = case["bins"]
    min_row_count = 1

    hist = FederatedHistogram(agg_client=DummyAggClient())
    result = hist.hist(
        data=df,
        y_var="y",
        x_vars=["group"],
        metadata=metadata,
        bins=bins,
        min_row_count=min_row_count,
    )

    y_values = df["y"].to_numpy(dtype=float)
    global_min = float(np.min(y_values))
    global_max = float(np.max(y_values))
    if global_min == global_max:
        global_max = global_min + 1.0
    bin_edges = np.linspace(global_min, global_max, bins + 1)
    expected_counts, _ = np.histogram(y_values, bins=bin_edges)
    expected_counts = [
        count if count >= min_row_count else None for count in expected_counts.tolist()
    ]

    assert result.bins == pytest.approx(bin_edges.tolist())
    assert result.counts == expected_counts

    grouped = result.grouped["group"]
    assert grouped.groups == ["A", "B", "C"]
    for group, counts in zip(grouped.groups, grouped.counts):
        subset = df.loc[df["group"] == group, "y"].to_numpy(dtype=float)
        expected_group_counts, _ = np.histogram(subset, bins=bin_edges)
        expected_group_counts = [
            count if count >= min_row_count else None
            for count in expected_group_counts.tolist()
        ]
        assert counts == expected_group_counts


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


@pytest.mark.parametrize("case", CATEGORICAL_CASES)
def test_histogram_categorical_matches_value_counts(case):
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

    min_row_count = 1

    hist = FederatedHistogram(agg_client=DummyAggClient())
    result = hist.hist(
        data=df,
        y_var="y",
        x_vars=["group"],
        metadata=metadata,
        bins=10,
        min_row_count=min_row_count,
    )

    expected_counts = [df["y"].value_counts().get(level, 0) for level in y_levels]
    expected_counts = [
        count if count >= min_row_count else None for count in expected_counts
    ]

    assert result.bins == y_levels
    assert result.counts == expected_counts

    grouped = result.grouped["group"]
    assert grouped.groups == ["A", "B", "C"]
    for group, counts in zip(grouped.groups, grouped.counts):
        subset = df.loc[df["group"] == group, "y"]
        expected_group_counts = [
            subset.value_counts().get(level, 0) for level in y_levels
        ]
        expected_group_counts = [
            count if count >= min_row_count else None for count in expected_group_counts
        ]
        assert counts == expected_group_counts
