from __future__ import annotations

import itertools
from functools import partial

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.sql.sql import FederatedSQL
from exaflow.algorithms.federated.sql.sql import FederatedSQLResults
from exaflow.algorithms.federated.statistics.primitive_statistics import (
    PrimitiveStatistics,
)
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)

BASIC_DATASETS = [
    {
        "name": "balanced_groups",
        "df": pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "x": [1.0, 3.0, 2.0, 4.0],
            }
        ),
    },
    {
        "name": "skewed_groups",
        "df": pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "C", "C"],
                "x": [1.0, 2.0, 8.0, 4.0, 10.0, 12.0],
            }
        ),
    },
    {
        "name": "null_group_values",
        "df": pd.DataFrame(
            {
                "group": ["A", None, "A", np.nan, "B"],
                "x": [2.0, 7.0, 4.0, 8.0, 6.0],
            }
        ),
    },
]


def _single_worker_aggregator() -> NumpyAggregator:
    coordinator = AggregationCoordinator(n_workers=1)
    client = SimulatedAggClient(worker_id=0, coordinator=coordinator)
    return NumpyAggregator(client)


def test_builder_exposes_only_allowed_next_steps():
    df = pd.DataFrame({"group": ["A", "B"], "x": [1.0, 2.0]})
    agg = _single_worker_aggregator()

    start = FederatedSQL(df, agg)
    assert hasattr(start, "where")
    assert hasattr(start, "group_by")
    assert hasattr(start, "aggregate")
    assert not hasattr(start, "run")

    after_where = start.where("x > 1.0")
    assert hasattr(after_where, "group_by")
    assert hasattr(after_where, "aggregate")
    assert not hasattr(after_where, "where")
    assert not hasattr(after_where, "run")

    after_group_by = after_where.group_by("group")
    assert hasattr(after_group_by, "aggregate")
    assert not hasattr(after_group_by, "group_by")
    assert not hasattr(after_group_by, "run")

    after_aggregate = after_group_by.aggregate("mean_x", agg.global_avg, "x")
    assert hasattr(after_aggregate, "aggregate")
    assert hasattr(after_aggregate, "run")
    assert not hasattr(after_aggregate, "where")
    assert not hasattr(after_aggregate, "group_by")


def test_run_returns_results_object():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    agg = _single_worker_aggregator()

    results = FederatedSQL(df, agg).aggregate("mean_x", agg.global_avg, "x").run()

    assert isinstance(results, FederatedSQLResults)
    assert list(results.dataframe.columns) == ["mean_x"]
    assert np.isclose(results.dataframe.loc[0, "mean_x"], 2.5)


def test_run_can_execute_only_once():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    agg = _single_worker_aggregator()
    query = FederatedSQL(df, agg).aggregate("sum_x", agg.global_sum, "x")

    first_results = query.run()
    assert np.isclose(first_results.dataframe.loc[0, "sum_x"], 6.0)

    with pytest.raises(RuntimeError, match="only be executed once"):
        query.run()


def test_multiple_aggregates_are_allowed():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    agg = _single_worker_aggregator()

    results = (
        FederatedSQL(df, agg)
        .aggregate("mean_x", agg.global_avg, "x")
        .aggregate("sum_x", agg.global_sum, "x")
        .run()
    )

    assert np.isclose(results.dataframe.loc[0, "mean_x"], 2.5)
    assert np.isclose(results.dataframe.loc[0, "sum_x"], 10.0)


class TestFederatedSQL(FederatedAlgorithmTest):
    def _split_inputs(self, X, y, n_workers: int):
        parts = np.array_split(np.arange(len(X)), n_workers)
        x_parts = [X.iloc[idx].reset_index(drop=True) for idx in parts]
        y_array = np.asarray(y)
        y_parts = [y_array[idx] for idx in parts]
        return x_parts, y_parts, X, y_array

    def compute_centralized_result(self, X, y, **kwargs):
        filtered = X.query("x >= 2.0")
        grouped = filtered.groupby("group", sort=True)["x"].mean()
        return grouped.to_dict()

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        aggregator = NumpyAggregator(agg_client)
        results = (
            FederatedSQL(X, aggregator)
            .where("x >= 2.0")
            .group_by("group")
            .aggregate("mean_x", aggregator.global_avg, "x")
            .run()
        )
        return results.dataframe["mean_x"].to_dict()

    def compare(self, federated_output, centralized_output, **kwargs):
        assert set(federated_output.keys()) == set(centralized_output.keys())
        for key in centralized_output:
            assert np.isclose(
                federated_output[key],
                centralized_output[key],
                atol=1e-7,
                rtol=1e-7,
                equal_nan=True,
            )

    @pytest.mark.parametrize(
        "dataset", BASIC_DATASETS, ids=[d["name"] for d in BASIC_DATASETS]
    )
    def test_federated_algorithm_with_one_worker(self, dataset):
        df = dataset["df"]
        self.run_comparison(
            X=df,
            y=np.zeros(len(df), dtype=float),
            n_workers=1,
        )

    @pytest.mark.parametrize(
        "dataset", BASIC_DATASETS, ids=[d["name"] for d in BASIC_DATASETS]
    )
    def test_federated_algorithm_with_multiple_workers(self, dataset):
        df = dataset["df"]
        self.run_comparison(
            X=df,
            y=np.zeros(len(df), dtype=float),
            n_workers=3,
        )


GROUP_BY_DATASETS = [
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
    {
        "name": "nulls_in_group_column",
        "df": pd.DataFrame(
            {
                "group": ["A", "A", None, np.nan, "B", "B"],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [2.0, 2.5, 3.5, 4.5, 5.5, 6.5],
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


def _centralized_per_group(df: pd.DataFrame, method: str, ddof: int = 0) -> dict:
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


def _federated_per_group(
    df_part: pd.DataFrame,
    agg_client,
    method: str,
    ddof: int = 0,
) -> dict:
    aggregator = NumpyAggregator(agg_client)
    ps = PrimitiveStatistics(aggregator)

    if method in ("mean", "variance", "standard_deviation", "covariance"):
        fn = partial(getattr(ps, method), ddof=ddof)
    else:
        fn = getattr(ps, method)

    cols = ["x"] if method in UNIVARIATE_METHODS else ["x", "y"]
    result_df = (
        FederatedSQL(df_part, aggregator)
        .group_by("group")
        .aggregate(method, fn, *cols)
        .run()
        .dataframe
    )
    return result_df[method].to_dict()


class TestFederatedSQLGroupByParity(FederatedAlgorithmTest):
    def _split_inputs(self, X, y, n_workers: int):
        parts = np.array_split(np.arange(len(X)), n_workers)
        x_parts = [X.iloc[idx].reset_index(drop=True) for idx in parts]
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
        return _federated_per_group(X, agg_client, method=method, ddof=ddof)

    def compare(self, federated_output: dict, centralized_output: dict, **kwargs):
        assert set(federated_output.keys()) == set(centralized_output.keys())
        for key in centralized_output:
            assert np.isclose(
                federated_output[key],
                centralized_output[key],
                atol=1e-7,
                rtol=1e-7,
                equal_nan=True,
            )

    @pytest.mark.parametrize(
        "dataset", GROUP_BY_DATASETS, ids=[d["name"] for d in GROUP_BY_DATASETS]
    )
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_federated_algorithm_with_one_worker(self, dataset, method):
        df = dataset["df"]
        self.run_comparison(
            X=df,
            y=np.zeros(len(df), dtype=float),
            n_workers=1,
            method=method,
            ddof=1 if method in ("variance", "standard_deviation", "covariance") else 0,
        )

    @pytest.mark.parametrize(
        "dataset", GROUP_BY_DATASETS, ids=[d["name"] for d in GROUP_BY_DATASETS]
    )
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_federated_algorithm_with_multiple_workers(self, dataset, method):
        df = dataset["df"]
        self.run_comparison(
            X=df,
            y=np.zeros(len(df), dtype=float),
            n_workers=3,
            method=method,
            ddof=1 if method in ("variance", "standard_deviation", "covariance") else 0,
        )


SKEWED_DATASETS = [
    {
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

SKEWED_DATASETS.append(
    {
        "name": "skewed_nulls_in_group",
        "partitions": [
            pd.DataFrame(
                {"group": ["A", "A", None], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}
            ),
            pd.DataFrame(
                {
                    "group": [np.nan, "B", "B"],
                    "x": [7.0, 8.0, 9.0],
                    "y": [1.0, 2.0, 3.0],
                }
            ),
        ],
    }
)


class TestSkewedFederatedSQL(FederatedAlgorithmTest):
    def _split_inputs(self, X, y, n_workers: int):
        if len(X) != n_workers:
            raise ValueError("The number of partitions must match n_workers.")
        x_parts = [part.reset_index(drop=True) for part in X]
        y_parts = [np.asarray([], dtype=float) for _ in x_parts]
        full_df = pd.concat(x_parts, ignore_index=True)
        full_y = np.asarray([], dtype=float)
        return x_parts, y_parts, full_df, full_y

    def compute_centralized_result(self, X, y, **kwargs):
        method = kwargs["method"]
        ddof = kwargs.get("ddof", 0)
        return _centralized_per_group(X, method=method, ddof=ddof)

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        method = kwargs["method"]
        ddof = kwargs.get("ddof", 0)
        return _federated_per_group(X, agg_client, method=method, ddof=ddof)

    def compare(self, federated_output, centralized_output, **kwargs):
        assert set(federated_output.keys()) == set(centralized_output.keys())
        for key in centralized_output:
            assert np.isclose(
                federated_output[key],
                centralized_output[key],
                atol=1e-7,
                rtol=1e-7,
                equal_nan=True,
            )

    @pytest.mark.parametrize(
        "dataset", SKEWED_DATASETS, ids=[d["name"] for d in SKEWED_DATASETS]
    )
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_federated_algorithm_with_one_worker(self, dataset, method):
        ddof = 1 if method in ("variance", "standard_deviation", "covariance") else 0
        merged = pd.concat(dataset["partitions"], ignore_index=True)
        self.run_comparison(
            X=[merged],
            y=np.asarray([], dtype=float),
            n_workers=1,
            method=method,
            ddof=ddof,
        )

    @pytest.mark.parametrize(
        "dataset", SKEWED_DATASETS, ids=[d["name"] for d in SKEWED_DATASETS]
    )
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_federated_algorithm_with_multiple_workers(self, dataset, method):
        ddof = 1 if method in ("variance", "standard_deviation", "covariance") else 0
        partitions = dataset["partitions"]
        self.run_comparison(
            X=partitions,
            y=np.asarray([], dtype=float),
            n_workers=len(partitions),
            method=method,
            ddof=ddof,
        )


class TestFederatedSQLPairwiseSMD(FederatedAlgorithmTest):
    def _split_inputs(self, X, y, n_workers: int):
        if len(X) != n_workers:
            raise ValueError("The number of partitions must match n_workers.")
        x_parts = [part.reset_index(drop=True) for part in X]
        y_parts = [np.asarray([], dtype=float) for _ in x_parts]
        full_df = pd.concat(x_parts, ignore_index=True)
        full_y = np.asarray([], dtype=float)
        return x_parts, y_parts, full_df, full_y

    def compute_centralized_result(self, X, y, **kwargs):
        g1 = kwargs["g1"]
        g2 = kwargs["g2"]
        slice_g1 = X[X["group"] == g1]["x"].to_numpy(dtype=float)
        slice_g2 = X[X["group"] == g2]["x"].to_numpy(dtype=float)

        s1 = slice_g1[~np.isnan(slice_g1)]
        s2 = slice_g2[~np.isnan(slice_g2)]
        n1, n2 = len(s1), len(s2)
        if n1 <= 1 or n2 <= 1:
            return np.nan

        mean1, mean2 = np.mean(s1), np.mean(s2)
        var1, var2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_sd if pooled_sd != 0 else 0.0

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        g1 = kwargs["g1"]
        g2 = kwargs["g2"]
        aggregator = NumpyAggregator(agg_client)
        ps = PrimitiveStatistics(aggregator)

        x1 = X[X["group"] == g1]["x"].to_numpy(dtype=float)
        x2 = X[X["group"] == g2]["x"].to_numpy(dtype=float)
        if len(x1) == 0:
            x1 = np.array([np.nan], dtype=float)
        if len(x2) == 0:
            x2 = np.array([np.nan], dtype=float)
        return ps.standardized_mean_differences(x1, x2)

    def compare(self, federated_output, centralized_output, **kwargs):
        assert np.isclose(
            federated_output,
            centralized_output,
            atol=1e-7,
            rtol=1e-7,
            equal_nan=True,
        )

    def _run_all_pairs(self, partitions):
        full_df = pd.concat(partitions, ignore_index=True)
        groups = sorted(full_df["group"].dropna().unique().tolist())
        n_workers = len(partitions)

        for g1, g2 in itertools.combinations(groups, 2):
            self.run_comparison(
                X=partitions,
                y=np.asarray([], dtype=float),
                n_workers=n_workers,
                g1=g1,
                g2=g2,
            )

    @pytest.mark.parametrize(
        "dataset", SKEWED_DATASETS, ids=[d["name"] for d in SKEWED_DATASETS]
    )
    def test_federated_algorithm_with_one_worker(self, dataset):
        merged = pd.concat(dataset["partitions"], ignore_index=True)
        self._run_all_pairs([merged])

    @pytest.mark.parametrize(
        "dataset", SKEWED_DATASETS, ids=[d["name"] for d in SKEWED_DATASETS]
    )
    def test_federated_algorithm_with_multiple_workers(self, dataset):
        self._run_all_pairs(dataset["partitions"])
