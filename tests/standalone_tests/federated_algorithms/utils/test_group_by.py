"""Tests for FederatedGroupBy.group_by() combined with PrimitiveStatistics.

Each test verifies that the per-group federated result matches the
equivalent pandas centralized group-by for every supported statistic.
"""

import itertools
from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from exaflow.algorithms.federated.statistics.primitive_statistics import (
    PrimitiveStatistics,
)
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from exaflow.algorithms.federated.utils.group_by import FederatedGroupBy
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    _simulate_federated_execution,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)

# ---------------------------------------------------------------------------
# Dataset definitions: each entry contains a DataFrame (X) and optional y.
# The DataFrame always has a "group" column used for the group-by.
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "name": "two_groups_balanced",
        "df": pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [2.0, 2.5, 3.5, 4.5, 5.5, 6.5],
            }
        ),
    },
    {
        "name": "three_groups_unequal",
        "df": pd.DataFrame(
            {
                "group": ["X", "X", "Y", "Y", "Y", "Z", "Z", "Z", "Z"],
                "x": [1.2, 1.8, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5],
                "y": [2.2, 2.8, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
            }
        ),
    },
    {
        "name": "negative_values",
        "df": pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "x": [-3.0, -1.0, 1.0, 2.0, 4.0, 6.0],
                "y": [5.0, 3.0, 1.0, -1.0, -3.0, -5.0],
            }
        ),
    },
]

UNIVARIATE_METHODS = [
    "count",
    "mean",
    "variance",
    "standard_deviation",
    "range",
    "coefficient_of_variation",
    "mean_absolute_deviation",
    "root_mean_square",
    "mean_square",
]

BIVARIATE_METHODS = [
    "covariance",
    "pearson_correlation",
    "standardized_mean_differences",
]

ALL_METHODS = UNIVARIATE_METHODS + BIVARIATE_METHODS


# ---------------------------------------------------------------------------
# Centralized reference implementations (per group)
# ---------------------------------------------------------------------------


def _centralized_per_group(df: pd.DataFrame, method: str, ddof: int = 0) -> dict:
    """Return {group_key: scalar} using nan-aware pandas/numpy for reference.

    Matches the behavior of PrimitiveStatistics where NaNs are ignored.
    """
    results = {}
    for group_key, sub in df.groupby("group", sort=True):
        x = sub["x"].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)

        n_x = np.sum(~np.isnan(x))
        n_y = np.sum(~np.isnan(y))
        avg_x = np.nanmean(x) if n_x > 0 else np.nan
        avg_y = np.nanmean(y) if n_y > 0 else np.nan

        if method == "count":
            results[group_key] = float(n_x)
        elif method == "mean":
            results[group_key] = float(avg_x)
        elif method == "variance":
            val = np.nansum((x - avg_x) ** 2) / (n_x - ddof) if n_x > ddof else 0.0
            results[group_key] = float(val)
        elif method == "standard_deviation":
            val = np.nansum((x - avg_x) ** 2) / (n_x - ddof) if n_x > ddof else 0.0
            results[group_key] = float(np.sqrt(val))
        elif method == "range":
            results[group_key] = float(np.nanmax(x) - np.nanmin(x)) if n_x > 0 else 0.0
        elif method == "coefficient_of_variation":
            std = np.sqrt(np.nansum((x - avg_x) ** 2) / n_x) if n_x > 0 else 0.0
            results[group_key] = float(std / avg_x) if avg_x != 0 else 0.0
        elif method == "mean_absolute_deviation":
            results[group_key] = (
                float(np.nanmean(np.abs(x - avg_x))) if n_x > 0 else 0.0
            )
        elif method == "root_mean_square":
            results[group_key] = float(np.sqrt(np.nanmean(x**2))) if n_x > 0 else 0.0
        elif method == "mean_square":
            results[group_key] = float(np.nanvar(x)) if n_x > 0 else 0.0
        elif method == "covariance":
            # global_sum((x-avg_x)*(y-avg_y)) / (n-ddof) where n is count(x)
            sum_prod = np.nansum((x - avg_x) * (y - avg_y))
            results[group_key] = float(sum_prod / (n_x - ddof)) if n_x > ddof else 0.0
        elif method == "pearson_correlation":
            if n_x <= 1:
                results[group_key] = 0.0
            else:
                cov = np.nansum((x - avg_x) * (y - avg_y)) / (n_x - 1)
                var_x = np.nansum((x - avg_x) ** 2) / (n_x - 1)
                var_y = np.nansum((y - avg_y) ** 2) / (n_x - 1)
                if var_x <= 0 or var_y <= 0:
                    results[group_key] = 0.0
                else:
                    results[group_key] = float(cov / (np.sqrt(var_x) * np.sqrt(var_y)))
        elif method == "standardized_mean_differences":
            var1 = np.nanvar(x, ddof=1) if n_x > 1 else 0.0
            var2 = np.nanvar(y, ddof=1) if n_y > 1 else 0.0
            pooled_sd = np.sqrt(((n_x - 1) * var1 + (n_y - 1) * var2) / (n_x + n_y - 2))
            results[group_key] = (
                float((avg_x - avg_y) / pooled_sd) if pooled_sd != 0 else 0.0
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    return results


# ---------------------------------------------------------------------------
# Federated per-group computation
# ---------------------------------------------------------------------------


def _federated_per_group(
    df_part: pd.DataFrame,
    agg_client,
    method: str,
    global_keys: list,
    ddof: int = 0,
) -> dict:
    """Compute the grouped statistic for a single worker's data partition.

    Uses FederatedGroupBy.__call__ which handles synchronization and group
    iteration internally.
    """
    aggregator = NumpyAggregator(agg_client)
    ps = PrimitiveStatistics(aggregator)
    gb = FederatedGroupBy(df_part, aggregator)

    # Wrap the aggregation function to include ddof if needed
    if method in ("mean", "variance", "standard_deviation", "covariance"):
        fn = partial(getattr(ps, method), ddof=ddof)
    else:
        fn = getattr(ps, method)

    # Determine columns for the aggregation
    if method in UNIVARIATE_METHODS:
        cols = ["x"]
    else:  # Bivariate
        cols = ["x", "y"]

    # Call the federated group-by. It will return a DataFrame indexed by "group".
    result_df = gb.group_by("group").aggregate(method, fn, *cols)()

    # Convert the result column to a dict {group: value} for comparison
    return result_df[method].to_dict()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestFederatedGroupBy(FederatedAlgorithmTest):
    """Compare federated group_by + PrimitiveStatistics against centralized pandas."""

    # ------------------------------------------------------------------
    # FederatedAlgorithmTest plumbing
    # ------------------------------------------------------------------

    def _split_inputs(self, X, y, n_workers: int):
        """Override: split the DataFrame row-wise across workers."""
        parts = np.array_split(np.arange(len(X)), n_workers)
        x_parts = [X.iloc[idx].reset_index(drop=True) for idx in parts]
        # y is unused beyond the interface; mirror the split
        y_array = np.asarray(y)
        y_parts = [y_array[idx] for idx in parts]
        return x_parts, y_parts, X, y_array

    def compute_centralized_result(self, X, y, **kwargs):
        method = kwargs["method"]
        ddof = kwargs.get("ddof", 0)
        return _centralized_per_group(X, method=method, ddof=ddof)

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        method = kwargs["method"]
        ddof = kwargs.get("ddof", 0)
        global_keys = kwargs["global_keys"]
        return _federated_per_group(
            X, agg_client, method=method, global_keys=global_keys, ddof=ddof
        )

    def compare(self, federated_output: dict, centralized_output: dict, **kwargs):
        assert set(federated_output.keys()) == set(centralized_output.keys()), (
            f"Group keys differ: {set(federated_output)} vs {set(centralized_output)}"
        )
        for key in centralized_output:
            fed_val = federated_output[key]
            cen_val = centralized_output[key]
            # Use equal_nan=True because we expect NaNs for empty groups
            assert np.isclose(fed_val, cen_val, atol=1e-7, rtol=1e-7, equal_nan=True), (
                f"Group '{key}': federated={fed_val}, centralized={cen_val}"
            )

    # ------------------------------------------------------------------
    # run_comparison override: inject global_keys into kwargs so workers
    # can iterate over all group labels even when their local slice is empty.
    # ------------------------------------------------------------------

    def run_comparison(self, *, X, y, n_workers: int = 5, **kwargs):
        global_keys = sorted(X["group"].unique().tolist())
        kwargs["global_keys"] = global_keys
        super().run_comparison(X=X, y=y, n_workers=n_workers, **kwargs)

    # mandatory abstract stubs (actual tests are parametrized below)
    def test_federated_algorithm_with_one_worker(self):
        pass

    def test_federated_algorithm_with_multiple_workers(self):
        pass

    # ------------------------------------------------------------------
    # Parametrized tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("dataset", DATASETS, ids=[d["name"] for d in DATASETS])
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_group_by_one_worker(self, dataset, method):
        df = dataset["df"]
        self.run_comparison(
            X=df,
            y=np.zeros(len(df), dtype=float),
            n_workers=1,
            method=method,
            ddof=1 if method in ("variance", "standard_deviation", "covariance") else 0,
        )

    @pytest.mark.parametrize("dataset", DATASETS, ids=[d["name"] for d in DATASETS])
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_group_by_multiple_workers(self, dataset, method):
        df = dataset["df"]
        self.run_comparison(
            X=df,
            y=np.zeros(len(df), dtype=float),
            n_workers=3,
            method=method,
            ddof=1 if method in ("variance", "standard_deviation", "covariance") else 0,
        )


# ---------------------------------------------------------------------------
# Skewed-distribution datasets
# Each entry defines explicit per-worker partitions so that at least one
# worker is entirely missing one or more group categories.
# ---------------------------------------------------------------------------

SKEWED_DATASETS = [
    {
        # Worker 0 has only group A; worker 1 has only group B.
        "name": "two_workers_disjoint_groups",
        "partitions": [
            pd.DataFrame(
                {"group": ["A", "A", "A"], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}
            ),
            pd.DataFrame(
                {"group": ["B", "B", "B"], "x": [7.0, 8.0, 9.0], "y": [1.0, 2.0, 3.0]}
            ),
        ],
    },
    {
        # Worker 0 has A and B; worker 1 has only B; worker 2 has only C.
        "name": "three_workers_partial_overlap",
        "partitions": [
            pd.DataFrame(
                {"group": ["A", "A", "B"], "x": [1.0, 2.0, 3.0], "y": [6.0, 5.0, 4.0]}
            ),
            pd.DataFrame({"group": ["B", "B"], "x": [4.0, 5.0], "y": [3.0, 2.0]}),
            pd.DataFrame(
                {"group": ["C", "C", "C"], "x": [6.0, 7.0, 8.0], "y": [1.0, 2.0, 3.0]}
            ),
        ],
    },
    {
        # Worker 0 has A, B, C; workers 1 and 2 have only A.
        "name": "one_worker_has_all_groups",
        "partitions": [
            pd.DataFrame(
                {
                    "group": ["A", "B", "B", "C", "C", "A"],
                    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "y": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                }
            ),
            pd.DataFrame({"group": ["A", "A"], "x": [7.0, 8.0], "y": [2.0, 3.0]}),
            pd.DataFrame({"group": ["A", "A"], "x": [9.0, 10.0], "y": [4.0, 5.0]}),
        ],
    },
    {
        # Negative values; worker 0 misses group B entirely.
        "name": "negative_values_missing_group",
        "partitions": [
            pd.DataFrame(
                {"group": ["A", "A", "A"], "x": [-3.0, -1.0, 1.0], "y": [5.0, 3.0, 1.0]}
            ),
            pd.DataFrame(
                {
                    "group": ["A", "B", "B"],
                    "x": [2.0, 4.0, 6.0],
                    "y": [-1.0, -3.0, -5.0],
                }
            ),
        ],
    },
]

# ---------------------------------------------------------------------------
# Complex 100-row dataset with nulls and skew
# ---------------------------------------------------------------------------

_rng = np.random.RandomState(42)
_ga = pd.DataFrame(
    {"group": ["A"] * 40, "x": _rng.normal(10, 2, 40), "y": _rng.normal(20, 5, 40)}
)
_gb = pd.DataFrame(
    {"group": ["B"] * 30, "x": _rng.normal(30, 10, 30), "y": _rng.normal(5, 2, 30)}
)
_gc = pd.DataFrame(
    {"group": ["C"] * 20, "x": _rng.uniform(0, 100, 20), "y": _rng.uniform(0, 100, 20)}
)
_gd = pd.DataFrame(
    {"group": ["D"] * 10, "x": _rng.exponential(15, 10), "y": _rng.exponential(30, 10)}
)
for _df in [_ga, _gb, _gc, _gd]:
    # 10% nulls in x and y
    _df.loc[_rng.rand(len(_df)) < 0.1, "x"] = np.nan
    _df.loc[_rng.rand(len(_df)) < 0.1, "y"] = np.nan

SKEWED_DATASETS.append(
    {
        "name": "complex_100_rows_nulls_skew",
        "partitions": [
            pd.concat([_ga.iloc[:30], _gb.iloc[:10]]),
            pd.concat([_ga.iloc[30:], _gb.iloc[10:], _gc.iloc[:10]]),
            pd.concat([_gc.iloc[10:], _gd]),
        ],
    }
)


class TestSkewedFederatedGroupBy:
    """Tests for FederatedGroupBy when one or more workers lack certain group keys.

    Unlike TestFederatedGroupBy, partitions here are hand-crafted so that at
    least one worker has no rows at all for a given group key. This exercises
    the NaN-sentinel fallback that makes empty-slice workers contribute nothing
    to the global aggregation.
    """

    def _run_skewed(self, partitions, method, ddof=0):
        """Run the federated computation with explicit per-worker partitions."""
        full_df = pd.concat(partitions, ignore_index=True)
        global_keys = sorted(full_df["group"].unique().tolist())
        n_workers = len(partitions)

        def worker_fn(worker_id, agg_client):
            return _federated_per_group(
                partitions[worker_id],
                agg_client,
                method=method,
                global_keys=global_keys,
                ddof=ddof,
            )

        outputs = _simulate_federated_execution(n_workers, worker_fn)

        # All workers must agree
        baseline = outputs[0]
        for other in outputs[1:]:
            assert set(baseline.keys()) == set(other.keys())
            for key in baseline:
                assert np.isclose(baseline[key], other[key], atol=1e-7, rtol=1e-7), (
                    f"Workers disagree on group '{key}': {baseline[key]} vs {other[key]}"
                )

        # Must match centralized result
        expected = _centralized_per_group(full_df, method=method, ddof=ddof)
        assert set(baseline.keys()) == set(expected.keys())
        for key in expected:
            assert np.isclose(baseline[key], expected[key], atol=1e-7, rtol=1e-7), (
                f"Group '{key}': federated={baseline[key]}, centralized={expected[key]}"
            )

    @pytest.mark.parametrize(
        "dataset", SKEWED_DATASETS, ids=[d["name"] for d in SKEWED_DATASETS]
    )
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_skewed_group_by(self, dataset, method):
        ddof = 1 if method in ("variance", "standard_deviation", "covariance") else 0
        self._run_skewed(dataset["partitions"], method=method, ddof=ddof)


class TestFederatedPairwiseSMD(FederatedAlgorithmTest):
    """Specifically tests SMD of every pair of group values in the dataset.

    This verifies that cross-group SMD matches centralized results even when
    groups are distributed across multiple workers.
    """

    def compute_centralized_result(self, X, y, **kwargs):
        # We handle loop inside the test method for pairwise
        pass

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        # We handle loop inside the test method for pairwise
        pass

    @pytest.mark.parametrize(
        "dataset", SKEWED_DATASETS, ids=[d["name"] for d in SKEWED_DATASETS]
    )
    def test_smd_all_pairs(self, dataset):
        partitions = dataset["partitions"]
        full_df = pd.concat(partitions, ignore_index=True)
        groups = sorted(full_df["group"].unique().tolist())
        n_workers = len(partitions)

        # Iterate through every unique pair of groups (A-B, A-C, B-C, etc.)
        for g1, g2 in itertools.combinations(groups, 2):

            def worker_fn(worker_id, agg_client):
                aggregator = NumpyAggregator(agg_client)
                ps = PrimitiveStatistics(aggregator)
                df_part = partitions[worker_id]

                # Local slices for the two groups. Note: SMD compares group 1 vs group 2.
                x1 = df_part[df_part["group"] == g1]["x"].to_numpy(dtype=float)
                x2 = df_part[df_part["group"] == g2]["x"].to_numpy(dtype=float)

                # Use NaN sentinels if local slice is empty to keep all workers in step
                if len(x1) == 0:
                    x1 = np.array([np.nan], dtype=float)
                if len(x2) == 0:
                    x2 = np.array([np.nan], dtype=float)

                return ps.standardized_mean_differences(x1, x2)

            outputs = _simulate_federated_execution(n_workers, worker_fn)

            # Centralized reference result for the same pair
            slice_g1 = full_df[full_df["group"] == g1]["x"].to_numpy(dtype=float)
            slice_g2 = full_df[full_df["group"] == g2]["x"].to_numpy(dtype=float)

            # Reference calculation (Matches PrimitiveStatistics logic)
            # Remove NaNs for the centralized reference calculation
            s1 = slice_g1[~np.isnan(slice_g1)]
            s2 = slice_g2[~np.isnan(slice_g2)]
            n1, n2 = len(s1), len(s2)

            if n1 <= 1 or n2 <= 1:
                expected = np.nan
            else:
                mean1, mean2 = np.mean(s1), np.mean(s2)
                var1, var2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
                pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                expected = (mean1 - mean2) / pooled_sd if pooled_sd != 0 else 0.0

            # Compare all federated outputs to expectation
            for fed_val in outputs:
                assert np.isclose(
                    fed_val, expected, atol=1e-7, rtol=1e-7, equal_nan=True
                ), (
                    f"Cross-group SMD('{g1}', '{g2}') failed. "
                    f"Federated={fed_val}, Centralized={expected}"
                )
