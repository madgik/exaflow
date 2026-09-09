import json

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.standardized_mean_difference import local_step
from exaflow.algorithms.federated.statistics.standardized_mean_difference import (
    RESULT_COLUMNS,
)
from exaflow.algorithms.federated.statistics.standardized_mean_difference import (
    FederatedStandardizedMeanDifference,
)
from tests.standalone_tests.federated_algorithms.utils import FederatedAlgorithmTest

# Test cases for group SMD
GROUP_SMD_TEST_CASES = [
    {
        "name": "two_groups_balanced",
        "data": {"group": ["A", "A", "B", "B"], "value": [1.0, 2.0, 3.0, 4.0]},
    },
    {
        "name": "two_groups_unequal_sizes",
        "data": {
            "group": ["A", "A", "A", "A", "B", "B"],
            "value": [1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
        },
    },
    {
        "name": "two_groups_large_diff",
        "data": {
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [10.0, 11.0, 12.0, 0.0, 1.0, 2.0],
        },
    },
    {
        "name": "two_groups_zero_diff",
        "data": {
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        },
    },
    {
        "name": "three_groups",
        "data": {
            "group": ["A", "A", "B", "B", "C", "C"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
    },
    {
        "name": "four_groups_one_insufficient",
        "data": {
            "group": ["A", "B", "B", "C", "C", "D", "D"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
    },
    {
        "name": "all_groups_insufficient",
        "data": {"group": ["A", "B", "C"], "value": [1.0, 2.0, 3.0]},
    },
    {
        "name": "single_group",
        "data": {"group": ["A", "A", "A"], "value": [1.0, 2.0, 3.0]},
    },
    {
        "name": "empty_dataframe",
        "data": {"group": [], "value": []},
    },
    {
        "name": "numeric_group_values",
        "data": {"group": [1, 1, 2, 2, 3, 3], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
    },
    {
        "name": "with_nan_values",
        "data": {"group": ["A", "A", "B", "B"], "value": [1.0, np.nan, 3.0, 4.0]},
    },
]

# Skewed datasets with pre-split partitions
SKEWED_GROUP_SMD_DATASETS = [
    {
        "name": "two_workers_disjoint_groups",
        "partitions": [
            pd.DataFrame({"group": ["A", "A", "A"], "value": [1.0, 2.0, 3.0]}),
            pd.DataFrame({"group": ["B", "B", "B"], "value": [4.0, 5.0, 6.0]}),
        ],
    },
    {
        "name": "three_workers_partial_overlap",
        "partitions": [
            pd.DataFrame({"group": ["A", "A", "B"], "value": [1.0, 2.0, 3.0]}),
            pd.DataFrame({"group": ["B", "B"], "value": [4.0, 5.0]}),
            pd.DataFrame({"group": ["C", "C", "C"], "value": [6.0, 7.0, 8.0]}),
        ],
    },
    {
        "name": "one_worker_has_all_groups",
        "partitions": [
            pd.DataFrame(
                {
                    "group": ["A", "B", "B", "C", "C", "A"],
                    "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                }
            ),
            pd.DataFrame({"group": ["A", "A"], "value": [7.0, 8.0]}),
            pd.DataFrame({"group": ["A", "A"], "value": [9.0, 10.0]}),
        ],
    },
    {
        "name": "negative_values_missing_group",
        "partitions": [
            pd.DataFrame({"group": ["A", "A", "A"], "value": [-3.0, -1.0, 1.0]}),
            pd.DataFrame({"group": ["A", "B", "B"], "value": [2.0, 4.0, 6.0]}),
        ],
    },
]


class TestFederatedSMDGroup(FederatedAlgorithmTest):
    """Compare group SMD and its sufficient statistics across partitions."""

    def _split_inputs(self, X, y, n_workers: int):
        if isinstance(X, list):
            if len(X) != n_workers:
                raise ValueError("Number of partitions must match n_workers")
            return (
                X,
                [np.zeros(len(part)) for part in X],
                pd.concat(X, ignore_index=True),
                np.zeros(sum(len(part) for part in X)),
            )
        return super()._split_inputs(X, y, n_workers)

    def compute_centralized_result(self, X, y, **kwargs):
        groups = sorted(X["group"].dropna().unique(), key=str)
        minimum_group_size = max(2, kwargs.get("minimum_group_size", 2))
        results = []
        for i, group1 in enumerate(groups):
            for group2 in groups[i + 1 :]:
                values1 = X.loc[X["group"] == group1, "value"].dropna()
                values2 = X.loc[X["group"] == group2, "value"].dropna()
                n1, n2 = len(values1), len(values2)
                if n1 < minimum_group_size or n2 < minimum_group_size:
                    continue
                mean1, mean2 = values1.mean(), values2.mean()
                var1, var2 = values1.var(ddof=1), values2.var(ddof=1)
                pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                results.append(
                    {
                        "group1": group1,
                        "group2": group2,
                        "n1": n1,
                        "n2": n2,
                        "mean1": mean1,
                        "mean2": mean2,
                        "var1": var1,
                        "var2": var2,
                        "smd": (mean1 - mean2) / pooled_sd if pooled_sd > 0 else 0.0,
                    }
                )
        return pd.DataFrame(results, columns=RESULT_COLUMNS)

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        smd = FederatedStandardizedMeanDifference(agg_client)
        return smd.pairwise_by_group(
            data=X,
            value_var="value",
            group_var="group",
            minimum_group_size=kwargs.get("minimum_group_size", 2),
        )

    def _validate_federated_outputs(self, federated_outputs):
        for output in federated_outputs[1:]:
            self.compare(output, federated_outputs[0])

    def compare(self, federated_output, centralized_output, **kwargs):
        assert list(federated_output.columns) == list(centralized_output.columns)
        assert len(federated_output) == len(centralized_output)
        if centralized_output.empty:
            return
        pd.testing.assert_frame_equal(
            federated_output[["group1", "group2"]].reset_index(drop=True),
            centralized_output[["group1", "group2"]].reset_index(drop=True),
            check_dtype=False,
        )
        numeric_columns = ["n1", "n2", "mean1", "mean2", "var1", "var2", "smd"]
        np.testing.assert_allclose(
            federated_output[numeric_columns].to_numpy(dtype=float),
            centralized_output[numeric_columns].to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-10,
        )

    @pytest.mark.parametrize(
        "case", GROUP_SMD_TEST_CASES, ids=[c["name"] for c in GROUP_SMD_TEST_CASES]
    )
    def test_federated_algorithm_with_one_worker(self, case):
        df = pd.DataFrame(case["data"])
        self.run_comparison(X=df, y=np.zeros(len(df)), n_workers=1)

    @pytest.mark.parametrize(
        "case", GROUP_SMD_TEST_CASES, ids=[c["name"] for c in GROUP_SMD_TEST_CASES]
    )
    def test_federated_algorithm_with_multiple_workers(self, case):
        df = pd.DataFrame(case["data"])
        self.run_comparison(X=df, y=np.zeros(len(df)), n_workers=3)

    def test_minimum_group_size_filters_small_groups(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "B", "C", "C", "C"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )
        self.run_comparison(
            X=df,
            y=np.zeros(len(df)),
            n_workers=2,
            minimum_group_size=3,
        )

    def test_minimum_group_size_cannot_disable_sample_variance_requirement(self):
        df = pd.DataFrame(
            {
                "group": ["A", "B", "B"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        self.run_comparison(
            X=df,
            y=np.zeros(len(df)),
            n_workers=2,
            minimum_group_size=1,
        )


class TestFederatedSMDGroupSkewed(TestFederatedSMDGroup):
    """Check disjoint groups and uneven partition sizes."""

    @pytest.mark.parametrize(
        "dataset",
        SKEWED_GROUP_SMD_DATASETS,
        ids=[d["name"] for d in SKEWED_GROUP_SMD_DATASETS],
    )
    def test_federated_algorithm_with_one_worker(self, dataset):
        df = pd.concat(dataset["partitions"], ignore_index=True)
        self.run_comparison(X=df, y=np.zeros(len(df)), n_workers=1)

    @pytest.mark.parametrize(
        "dataset",
        SKEWED_GROUP_SMD_DATASETS,
        ids=[d["name"] for d in SKEWED_GROUP_SMD_DATASETS],
    )
    def test_federated_algorithm_with_multiple_workers(self, dataset):
        partitions = dataset["partitions"]
        self.run_comparison(X=partitions, y=[], n_workers=len(partitions))


class _ProductionShapedAggregationClient:
    """Return one-element arrays for scalar aggregations, like the gRPC client."""

    def sum(self, values):
        return np.asarray(values).reshape(-1)

    def union(self, values):
        return list(values)


def test_exareme3_result_contains_only_json_safe_pairwise_smd_values():
    data = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "value": np.arange(20, dtype=float),
        }
    )

    comparisons = local_step(
        _ProductionShapedAggregationClient(),
        data,
        group_var="group",
        value_var="value",
    )

    assert len(comparisons) == 1
    assert set(comparisons[0]) == {"group1", "group2", "smd"}
    assert comparisons[0]["group1"] == "A"
    assert comparisons[0]["group2"] == "B"
    assert isinstance(comparisons[0]["smd"], float)
    json.dumps(comparisons)
