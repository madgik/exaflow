from __future__ import annotations

from collections import Counter
from typing import Dict
from typing import Iterable
from typing import List

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.statistics.percentile import Percentile
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from exaflow.column_names import DATASET_COL


class DescribeResult:
    def __init__(
        self,
        recs: List[Dict],
        global_recs: List[Dict],
    ):
        self.recs = recs
        self.global_recs = global_recs


class FederatedDescribe:
    """
    Federated descriptive statistics algorithm.

    ``describe`` returns per-feature summaries where each feature is
    computed with pairwise complete observations.

    Notes
    -----
    - Per-dataset quantiles (q1/q2/q3) are exact (pandas). The global summary
      carries federated ``q1``/``q2``/``q3`` and ``median`` (= ``q2``) estimated
      across all datasets via a histogram-based percentile.
    - Means and variances come from aggregated sufficient statistics (sx, sxx).
    - Nominal counts flow through the aggregation client and zero-count levels are
      dropped in the global result per our contract.
    - NA handling remains pairwise complete per variable.
    - This is descriptive only—no inference or modeling.
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client

    def describe(
        self,
        *,
        data: pd.DataFrame,
        numerical_vars: List[str],
        nominal_vars: List[str],
        min_row_count: int,
        nominal_levels: Dict[str, List],
        dataset_col: str = DATASET_COL,
        integer_vars: List[str] | None = None,
    ) -> DescribeResult:
        datasets: Iterable[str]
        if dataset_col in data.columns:
            datasets = set(data[dataset_col])
        else:
            datasets = ["unknown"]

        recs = [
            rec
            for dataset in datasets
            for rec in self._compute_stats_records(
                df=(
                    data[data[dataset_col] == dataset]
                    if dataset_col in data.columns
                    else data
                ),
                dataset=str(dataset),
                numerical_vars=numerical_vars,
                nominal_vars=nominal_vars,
                min_row_count=min_row_count,
            )
        ]

        global_recs = self._aggregate_global(
            records=recs,
            variables=numerical_vars + nominal_vars,
            nominal_levels=nominal_levels,
        )

        self._add_global_quartiles(
            global_recs=global_recs,
            recs=recs,
            data=data,
            numerical_vars=numerical_vars,
            dataset_col=dataset_col,
            integer_vars=integer_vars or [],
        )

        return DescribeResult(
            recs=recs,
            global_recs=global_recs,
        )

    def _add_global_quartiles(
        self,
        *,
        global_recs: List[Dict],
        recs: List[Dict],
        data: pd.DataFrame,
        numerical_vars: List[str],
        dataset_col: str,
        integer_vars: List[str],
    ) -> None:
        """Estimate federated q1/q2/q3 and median per numerical variable across
        all datasets and attach them to the global summary record.

        Unlike the per-dataset ``q1``/``q2``/``q3`` (exact, computed locally with
        pandas), these are histogram-based estimates aggregated across every
        worker — true global quartiles rather than local ones. ``median`` is set
        to the same value as ``q2`` for readability.

        The estimate is restricted to rows from ``(variable, dataset)`` groups
        that cleared the ``min_row_count`` threshold (those with a non-null
        per-dataset record), matching the population used for the other global
        statistics. Otherwise rows withheld at the dataset level for privacy
        would still influence the published quartiles.

        Every worker iterates ``numerical_vars`` in the same order and the
        estimator's federated calls are identical across workers, so the
        aggregation stays in lock-step even when a worker holds no rows for a
        variable.
        """
        if not numerical_vars:
            return
        estimator = Percentile(NumpyAggregator(self.agg_client))
        recs_by_var = {rec["variable"]: rec for rec in global_recs}
        integer_var_set = set(integer_vars)
        has_dataset_col = dataset_col in data.columns
        for var in numerical_vars:
            is_integer = var in integer_var_set
            allowed = {
                rec["dataset"]
                for rec in recs
                if rec["variable"] == var and rec["data"] is not None
            }
            if has_dataset_col:
                col = data.loc[data[dataset_col].astype(str).isin(allowed), var]
            else:
                col = data[var] if allowed else data[var].iloc[0:0]
            values = {}
            for key, q in [("q1", 0.25), ("q2", 0.5), ("q3", 0.75)]:
                result = estimator.compute(col, q, is_integer=is_integer)
                values[key] = None if result is None else float(result[0])
            rec = recs_by_var.get(var)
            if rec is None or rec["data"] is None:
                continue
            rec["data"]["q1"] = values["q1"]
            rec["data"]["q2"] = values["q2"]
            rec["data"]["q3"] = values["q3"]
            rec["data"]["median"] = values["q2"]

    def _compute_stats_records(
        self,
        *,
        df: pd.DataFrame,
        dataset: str,
        numerical_vars: List[str],
        nominal_vars: List[str],
        min_row_count: int,
    ) -> List[Dict]:
        def record(var, dataset_name, payload):
            return dict(variable=var, dataset=dataset_name, data=payload)

        variables = numerical_vars + nominal_vars
        if len(df) < min_row_count:
            return [record(var, dataset, None) for var in variables]

        num_total = int(len(df))
        descr_all = df.describe(include="all")
        descr_numerical = (
            df[numerical_vars].describe() if numerical_vars else pd.DataFrame()
        )
        num_dtps = descr_all.loc["count"].to_dict()
        num_na = {var: num_total - num_dtps.get(var, 0) for var in variables}

        numerical_recs = []
        if numerical_vars:
            sx = df[numerical_vars].sum().to_dict()
            sxx = (df[numerical_vars] ** 2).sum().to_dict()
            q1 = descr_numerical.loc["25%"].to_dict()
            q2 = descr_numerical.loc["50%"].to_dict()
            q3 = descr_numerical.loc["75%"].to_dict()
            means = descr_numerical.loc["mean"].to_dict()
            stds = descr_numerical.loc["std"].to_dict()
            mins = descr_numerical.loc["min"].to_dict()
            maxs = descr_numerical.loc["max"].to_dict()

            for var in numerical_vars:
                if num_dtps.get(var, 0) < min_row_count:
                    numerical_recs.append(record(var, dataset, None))
                    continue
                local_std = stds.get(var)
                if local_std is not None and np.isnan(local_std):
                    local_std = None
                numerical_recs.append(
                    record(
                        var,
                        dataset,
                        dict(
                            num_dtps=int(num_dtps[var]),
                            num_na=int(num_na[var]),
                            num_total=num_total,
                            sx=float(sx[var]),
                            sxx=float(sxx[var]),
                            q1=float(q1[var]),
                            q2=float(q2[var]),
                            q3=float(q3[var]),
                            mean=float(means[var]),
                            std=local_std,
                            min=float(mins[var]),
                            max=float(maxs[var]),
                        ),
                    )
                )

        nominal_recs = []
        for var in nominal_vars:
            if num_dtps.get(var, 0) < min_row_count:
                nominal_recs.append(record(var, dataset, None))
                continue
            nominal_recs.append(
                record(
                    var,
                    dataset,
                    dict(
                        num_dtps=int(num_dtps[var]),
                        num_na=int(num_na[var]),
                        num_total=num_total,
                        counts=df[var].value_counts().to_dict(),
                    ),
                )
            )

        return numerical_recs + nominal_recs

    def _aggregate_global(
        self,
        *,
        records: List[Dict],
        variables: List[str],
        nominal_levels: Dict[str, List],
        dataset_label: str = "all datasets",
    ) -> List[Dict]:
        valid = [r for r in records if r.get("data") is not None]

        numerical_vars = [v for v in variables if v not in nominal_levels]
        nominal_vars = [v for v in variables if v in nominal_levels]

        num_stats = {
            v: {
                "num_dtps": 0.0,
                "num_na": 0.0,
                "num_total": 0.0,
                "sx": 0.0,
                "sxx": 0.0,
                "min": np.inf,
                "max": -np.inf,
            }
            for v in numerical_vars
        }
        cat_stats = {
            v: {"num_dtps": 0.0, "num_na": 0.0, "num_total": 0.0, "counts": Counter()}
            for v in nominal_vars
        }

        for rec in valid:
            var = rec["variable"]
            data = rec["data"]
            if var in num_stats:
                stats = num_stats[var]
                stats["num_dtps"] += data["num_dtps"]
                stats["num_na"] += data["num_na"]
                stats["num_total"] += data["num_total"]
                stats["sx"] += data["sx"]
                stats["sxx"] += data["sxx"]
                stats["min"] = min(stats["min"], data["min"])
                stats["max"] = max(stats["max"], data["max"])
            elif var in cat_stats:
                stats = cat_stats[var]
                stats["num_dtps"] += data["num_dtps"]
                stats["num_na"] += data["num_na"]
                stats["num_total"] += data["num_total"]
                stats["counts"] += Counter(data["counts"])

        num_dtps_arr = np.array(
            [num_stats[v]["num_dtps"] for v in numerical_vars], dtype=float
        )
        num_na_arr = np.array(
            [num_stats[v]["num_na"] for v in numerical_vars], dtype=float
        )
        num_total_arr = np.array(
            [num_stats[v]["num_total"] for v in numerical_vars], dtype=float
        )
        sx_arr = np.array([num_stats[v]["sx"] for v in numerical_vars], dtype=float)
        sxx_arr = np.array([num_stats[v]["sxx"] for v in numerical_vars], dtype=float)
        min_arr = np.array([num_stats[v]["min"] for v in numerical_vars], dtype=float)
        max_arr = np.array([num_stats[v]["max"] for v in numerical_vars], dtype=float)

        num_dtps_arr = np.asarray(self.agg_client.sum(num_dtps_arr), dtype=float)
        num_na_arr = np.asarray(self.agg_client.sum(num_na_arr), dtype=float)
        num_total_arr = np.asarray(self.agg_client.sum(num_total_arr), dtype=float)
        sx_arr = np.asarray(self.agg_client.sum(sx_arr), dtype=float)
        sxx_arr = np.asarray(self.agg_client.sum(sxx_arr), dtype=float)
        min_arr = np.asarray(self.agg_client.min(min_arr), dtype=float)
        max_arr = np.asarray(self.agg_client.max(max_arr), dtype=float)

        cat_num_dtps = np.array(
            [cat_stats[v]["num_dtps"] for v in nominal_vars], dtype=float
        )
        cat_num_na = np.array(
            [cat_stats[v]["num_na"] for v in nominal_vars], dtype=float
        )
        cat_num_total = np.array(
            [cat_stats[v]["num_total"] for v in nominal_vars], dtype=float
        )

        cat_num_dtps = np.asarray(self.agg_client.sum(cat_num_dtps), dtype=float)
        cat_num_na = np.asarray(self.agg_client.sum(cat_num_na), dtype=float)
        cat_num_total = np.asarray(self.agg_client.sum(cat_num_total), dtype=float)

        global_records: List[Dict] = []
        for idx, var in enumerate(numerical_vars):
            num_dtps = float(num_dtps_arr[idx]) if numerical_vars else 0.0
            if num_dtps <= 0:
                global_records.append(
                    dict(variable=var, dataset=dataset_label, data=None)
                )
                continue
            mean = float(sx_arr[idx]) / num_dtps
            if num_dtps <= 1:
                std = None
            else:
                variance = (float(sxx_arr[idx]) - num_dtps * (mean**2)) / (num_dtps - 1)
                std = float(np.sqrt(max(variance, 0.0)))
            global_records.append(
                dict(
                    variable=var,
                    dataset=dataset_label,
                    data=dict(
                        num_dtps=int(num_dtps),
                        num_na=int(num_na_arr[idx]),
                        num_total=int(num_total_arr[idx]),
                        mean=mean,
                        std=std,
                        min=float(min_arr[idx]),
                        max=float(max_arr[idx]),
                    ),
                )
            )

        for idx, var in enumerate(nominal_vars):
            num_dtps = float(cat_num_dtps[idx]) if nominal_vars else 0.0
            if num_dtps <= 0:
                global_records.append(
                    dict(variable=var, dataset=dataset_label, data=None)
                )
                continue
            levels = nominal_levels[var]
            local_counts = np.array(
                [cat_stats[var]["counts"].get(level, 0) for level in levels],
                dtype=float,
            )
            counts_arr = np.asarray(self.agg_client.sum(local_counts), dtype=float)
            counts = {
                level: int(counts_arr[i])
                for i, level in enumerate(levels)
                if counts_arr[i] > 0
            }
            global_records.append(
                dict(
                    variable=var,
                    dataset=dataset_label,
                    data=dict(
                        num_dtps=int(num_dtps),
                        num_na=int(cat_num_na[idx]),
                        num_total=int(cat_num_total[idx]),
                        counts=counts,
                    ),
                )
            )

        return global_records
