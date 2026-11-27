from __future__ import annotations

from collections import Counter
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.statistics.percentile import Percentile
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from exaflow.algorithms.federated.validation import check_dataframe


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
    - Per-group quantiles (q1/q2/q3) are exact (pandas). The global summary
      carries federated ``q1``/``q2``/``q3`` and ``median`` (= ``q2``) estimated
      across all groups via a histogram-based percentile.
    - Means and variances come from aggregated counts and centered moments.
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
        group_column: str,
        integer_vars: List[str] | None = None,
    ) -> DescribeResult:
        """Summarize rows by the values in a required grouping column."""
        check_dataframe(data, [group_column])
        group_values = data[group_column]

        recs = [
            rec
            for group in set(group_values)
            for rec in self._compute_stats_records(
                df=data[group_values == group],
                group=group,
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
            group_values=group_values,
            integer_vars=integer_vars or [],
        )

        return DescribeResult(
            recs=[dict(rec, group=str(rec["group"])) for rec in recs],
            global_recs=global_recs,
        )

    def _add_global_quartiles(
        self,
        *,
        global_recs: List[Dict],
        recs: List[Dict],
        data: pd.DataFrame,
        numerical_vars: List[str],
        group_values: pd.Series,
        integer_vars: List[str],
    ) -> None:
        """Estimate federated q1/q2/q3 and median per numerical variable across
        all groups and attach them to the global summary record.

        Unlike the per-group ``q1``/``q2``/``q3`` (exact, computed locally with
        pandas), these are histogram-based estimates aggregated across every
        worker — true global quartiles rather than local ones. ``median`` is set
        to the same value as ``q2`` for readability.

        The estimate is restricted to rows from ``(variable, group)`` pairs
        that cleared the ``min_row_count`` threshold (those with a non-null
        per-group record), matching the population used for the other global
        statistics. Otherwise rows withheld at the group level for privacy
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
        for var in numerical_vars:
            is_integer = var in integer_var_set
            allowed = {
                rec["group"]
                for rec in recs
                if rec["variable"] == var and rec["data"] is not None
            }
            col = data.loc[group_values.isin(allowed), var]
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
        group: object,
        numerical_vars: List[str],
        nominal_vars: List[str],
        min_row_count: int,
    ) -> List[Dict]:
        def record(var, group_label, payload):
            return dict(variable=var, group=group_label, data=payload)

        variables = numerical_vars + nominal_vars
        if len(df) < min_row_count:
            return [record(var, group, None) for var in variables]

        num_total = int(len(df))
        descr_all = df.describe(include="all")
        descr_numerical = (
            df[numerical_vars].astype(np.float64).describe()
            if numerical_vars
            else pd.DataFrame()
        )
        num_dtps = descr_all.loc["count"].to_dict()
        num_na = {var: num_total - num_dtps.get(var, 0) for var in variables}

        numerical_recs = []
        if numerical_vars:
            q1 = descr_numerical.loc["25%"].to_dict()
            q2 = descr_numerical.loc["50%"].to_dict()
            q3 = descr_numerical.loc["75%"].to_dict()
            means = descr_numerical.loc["mean"].to_dict()
            stds = descr_numerical.loc["std"].to_dict()
            mins = descr_numerical.loc["min"].to_dict()
            maxs = descr_numerical.loc["max"].to_dict()

            for var in numerical_vars:
                count = int(num_dtps[var])
                if count < min_row_count:
                    numerical_recs.append(record(var, group, None))
                    continue
                local_std = stds.get(var)
                if local_std is not None and np.isnan(local_std):
                    local_std = None
                numerical_recs.append(
                    record(
                        var,
                        group,
                        dict(
                            num_dtps=count,
                            num_na=int(num_na[var]),
                            num_total=num_total,
                            sx=float(means[var]) * count if count else 0.0,
                            m2=(
                                (count - 1) * float(local_std) ** 2
                                if local_std is not None
                                else 0.0
                            ),
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
                nominal_recs.append(record(var, group, None))
                continue
            nominal_recs.append(
                record(
                    var,
                    group,
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
                "moments": [],
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
                if data["num_dtps"] == 0:
                    continue
                stats["sx"] += data["sx"]
                stats["moments"].append((data["num_dtps"], data["mean"], data["m2"]))
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
        min_arr = np.array([num_stats[v]["min"] for v in numerical_vars], dtype=float)
        max_arr = np.array([num_stats[v]["max"] for v in numerical_vars], dtype=float)

        num_dtps_arr = np.asarray(self.agg_client.sum(num_dtps_arr), dtype=float)
        num_na_arr = np.asarray(self.agg_client.sum(num_na_arr), dtype=float)
        num_total_arr = np.asarray(self.agg_client.sum(num_total_arr), dtype=float)
        sx_arr = np.asarray(self.agg_client.sum(sx_arr), dtype=float)
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

        means = np.divide(
            sx_arr, num_dtps_arr, out=np.zeros_like(sx_arr), where=num_dtps_arr > 0
        )
        local_m2 = np.array(
            [
                sum(
                    m2 + n * (mean - means[idx]) ** 2
                    for n, mean, m2 in num_stats[var]["moments"]
                )
                for idx, var in enumerate(numerical_vars)
            ],
            dtype=float,
        )
        global_m2 = np.asarray(self.agg_client.sum(local_m2), dtype=float)

        global_records: List[Dict] = []
        for idx, var in enumerate(numerical_vars):
            num_dtps = float(num_dtps_arr[idx]) if numerical_vars else 0.0
            if num_dtps <= 0:
                global_records.append(dict(variable=var, group=None, data=None))
                continue
            mean = float(means[idx])
            if num_dtps <= 1:
                std = None
            else:
                std = float(np.sqrt(global_m2[idx] / (num_dtps - 1)))
            global_records.append(
                dict(
                    variable=var,
                    group=None,
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
                global_records.append(dict(variable=var, group=None, data=None))
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
                    group=None,
                    data=dict(
                        num_dtps=int(num_dtps),
                        num_na=int(cat_num_na[idx]),
                        num_total=int(cat_num_total[idx]),
                        counts=counts,
                    ),
                )
            )

        return global_records
