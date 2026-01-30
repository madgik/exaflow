from __future__ import annotations

import itertools
from typing import Iterable
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.libqsturng import psturng


class FederatedAnovaOneWay:
    """
    Federated one-way ANOVA with a statsmodels-like interface.

    This class mirrors the behavior of statsmodels' one-way ANOVA utilities
    while computing sufficient statistics via secure aggregation.
    """

    def __init__(self, agg_client, *, alpha: float = 0.05):
        """
        Initialize a federated one-way ANOVA estimator.

        Parameters
        ----------
        agg_client
            Aggregation client used to sum local sufficient statistics.
        alpha
            Significance level used for Tukey HSD p-values.
        """
        self.agg_client = agg_client
        self.alpha = alpha

    def fit(self, groups: Sequence[Iterable[float]], categories: Sequence):
        """
        Fit one-way ANOVA from per-group observations.

        Parameters
        ----------
        groups
            Sequence of 1D arrays/lists, each containing the observations for
            a group. The order must match `categories`.
        categories
            Labels for each group in `groups`.
        """
        stats = self._collect_stats(groups, categories)

        group_stats_count = stats["group_stats_count"]
        group_stats_sum = stats["group_stats_sum"]
        group_ssq = stats["group_ssq"]
        var_min_per_group = stats["var_min_per_group"]
        var_max_per_group = stats["var_max_per_group"]
        n_obs = stats["n_obs"]
        overall_sum = stats["overall_sum"]
        overall_count = stats["overall_count"]
        overall_ssq = stats["overall_ssq"]
        group_stats_index = stats["group_stats_index"]

        if len(group_stats_index) < 2:
            raise ValueError(
                "Cannot perform Anova one-way. Covariable has only one level."
            )

        df_explained = len(group_stats_index) - 1
        df_residual = n_obs - len(group_stats_index)

        overall_mean = overall_sum / overall_count if overall_count else 0.0
        ss_residual = overall_ssq - np.sum(group_stats_sum**2 / group_stats_count)
        ss_explained = np.sum(
            (overall_mean - group_stats_sum / group_stats_count) ** 2
            * group_stats_count
        )

        ms_explained = ss_explained / df_explained
        ms_residual = ss_residual / df_residual
        f_stat = ms_explained / ms_residual if ms_residual != 0 else 0.0
        p_value = 1.0 - st.f.cdf(f_stat, df_explained, df_residual)

        self.table_ = pd.DataFrame(
            [
                [df_explained, ss_explained, ms_explained, f_stat, p_value],
                [df_residual, ss_residual, ms_residual, None, None],
            ],
            columns=["df", "sum_sq", "mean_sq", "F", "PR(>F)"],
            index=("categories", "Residual"),
        )

        g_cat = np.array(group_stats_index)
        n_groups = len(g_cat)
        gnobs = group_stats_count
        gmeans = group_stats_sum / gnobs
        gvar = self.table_.loc["Residual"]["mean_sq"] / gnobs
        g1, g2 = np.array(list(itertools.combinations(np.arange(n_groups), 2))).T

        mn = gmeans[g1] - gmeans[g2]
        se = np.sqrt(gvar[g1] + gvar[g2])
        tval = mn / se
        df = self.table_.at["Residual", "df"]
        pval = psturng(np.sqrt(2.0) * np.abs(tval), n_groups, df)

        thsd = pd.DataFrame(
            columns=[
                "A",
                "B",
                "mean(A)",
                "mean(B)",
                "diff",
                "Std.Err.",
                "t value",
                "Pr(>|t|)",
            ],
            index=range(n_groups * (n_groups - 1) // 2),
        )
        thsd["A"] = np.array(g_cat)[g1.astype(int)]
        thsd["B"] = np.array(g_cat)[g2.astype(int)]
        thsd["mean(A)"] = gmeans[g1]
        thsd["mean(B)"] = gmeans[g2]
        thsd["diff"] = mn
        thsd["Std.Err."] = se
        thsd["t value"] = tval
        thsd["Pr(>|t|)"] = pval

        variances = group_ssq / group_stats_count - gmeans**2
        sample_vars = (group_stats_count - 1.0) / group_stats_count * variances
        sample_stds = np.sqrt(sample_vars)

        self.nobs = int(n_obs)
        self.fvalue = float(f_stat)
        self.pvalue = float(p_value)
        self.df_between = float(df_explained)
        self.df_within = float(df_residual)
        self.ss_between = float(ss_explained)
        self.ss_within = float(ss_residual)
        self.ms_between = float(ms_explained)
        self.ms_within = float(ms_residual)
        self.categories_ = list(categories)
        self.group_stats_index_ = list(group_stats_index)
        self.means_ = gmeans.tolist()
        self.sample_stds_ = sample_stds.tolist()
        self.var_min_per_group_ = var_min_per_group.tolist()
        self.var_max_per_group_ = var_max_per_group.tolist()
        self.thsd_ = thsd

        return self

    def _collect_stats(self, groups: Sequence[Iterable[float]], categories: Sequence):
        """
        Aggregate group-level sufficient statistics across workers.

        Each group corresponds to one category in `categories`.
        """
        group_stats_count_local = []
        group_stats_sum_local = []
        group_ssq_local = []
        var_min_local = []
        var_max_local = []

        for values in groups:
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == 0:
                group_stats_count_local.append(0.0)
                group_stats_sum_local.append(0.0)
                group_ssq_local.append(0.0)
                var_min_local.append(np.inf)
                var_max_local.append(-np.inf)
            else:
                group_stats_count_local.append(float(arr.size))
                group_stats_sum_local.append(float(arr.sum()))
                group_ssq_local.append(float((arr**2).sum()))
                var_min_local.append(float(arr.min()))
                var_max_local.append(float(arr.max()))

        group_stats_count_local = np.asarray(group_stats_count_local, dtype=float)
        group_stats_sum_local = np.asarray(group_stats_sum_local, dtype=float)
        group_ssq_local = np.asarray(group_ssq_local, dtype=float)
        var_min_local = np.asarray(var_min_local, dtype=float)
        var_max_local = np.asarray(var_max_local, dtype=float)

        n_obs_local = float(group_stats_count_local.sum())
        overall_sum_local = float(group_stats_sum_local.sum())
        overall_count_local = float(group_stats_count_local.sum())
        overall_ssq_local = float(group_ssq_local.sum())

        total_n_obs_arr = self.agg_client.sum(np.array([n_obs_local], dtype=float))
        total_overall_sum_arr = self.agg_client.sum(
            np.array([overall_sum_local], dtype=float)
        )
        total_overall_count_arr = self.agg_client.sum(
            np.array([overall_count_local], dtype=float)
        )
        total_overall_ssq_arr = self.agg_client.sum(
            np.array([overall_ssq_local], dtype=float)
        )
        group_stats_sum_arr = self.agg_client.sum(group_stats_sum_local)
        group_stats_count_arr = self.agg_client.sum(group_stats_count_local)
        group_ssq_arr = self.agg_client.sum(group_ssq_local)
        var_min_arr = self.agg_client.min(var_min_local)
        var_max_arr = self.agg_client.max(var_max_local)

        n_obs = int(np.asarray(total_n_obs_arr).reshape(-1)[0])
        overall_sum = float(np.asarray(total_overall_sum_arr).reshape(-1)[0])
        overall_count = float(np.asarray(total_overall_count_arr).reshape(-1)[0])
        overall_ssq = float(np.asarray(total_overall_ssq_arr).reshape(-1)[0])
        group_stats_sum = np.asarray(group_stats_sum_arr, dtype=float)
        group_stats_count = np.asarray(group_stats_count_arr, dtype=float)
        group_ssq = np.asarray(group_ssq_arr, dtype=float)
        var_min_per_group = np.asarray(var_min_arr, dtype=float)
        var_max_per_group = np.asarray(var_max_arr, dtype=float)

        nonzero_mask = group_stats_count != 0
        if not np.all(nonzero_mask):
            group_stats_count = group_stats_count[nonzero_mask]
            group_stats_sum = group_stats_sum[nonzero_mask]
            group_ssq = group_ssq[nonzero_mask]
            var_min_per_group = var_min_per_group[nonzero_mask]
            var_max_per_group = var_max_per_group[nonzero_mask]
            group_stats_index = [c for c, m in zip(categories, nonzero_mask) if m]
        else:
            group_stats_index = list(categories)

        return {
            "n_obs": n_obs,
            "overall_sum": overall_sum,
            "overall_count": overall_count,
            "overall_ssq": overall_ssq,
            "group_stats_sum": group_stats_sum,
            "group_stats_count": group_stats_count,
            "group_ssq": group_ssq,
            "var_min_per_group": var_min_per_group,
            "var_max_per_group": var_max_per_group,
            "group_stats_index": group_stats_index,
        }
