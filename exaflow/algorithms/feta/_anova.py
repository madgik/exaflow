"""FETA-owned ANOVA inference and output formatting from global summaries."""

import itertools
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.libqsturng import psturng

from exaflow.worker_communication import BadUserInput


class AnovaStatistics:
    def fit(self, stats, categories):
        """Compute inference from completed global summaries, without worker calls."""
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
            raise BadUserInput(
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


def get_min_max_ci_info(
    *,
    means: List[float],
    sample_stds: List[float],
    categories: List[str],
    var_min_per_group: List[float],
    var_max_per_group: List[float],
    group_stats_index: List[str],
) -> Tuple[Dict, Dict]:
    """Builds min/max per group and confidence interval information tables."""
    aligned_categories = [c for c in categories if c in group_stats_index]

    ci_df = pd.DataFrame(
        {
            "means": list(means),
            "sample_stds": list(sample_stds),
        },
        index=aligned_categories,
    )
    ci_df["m-s"] = ci_df["means"] - ci_df["sample_stds"]
    ci_df["m+s"] = ci_df["means"] + ci_df["sample_stds"]

    min_max_per_group = {
        "categories": aligned_categories,
        "min": var_min_per_group,
        "max": var_max_per_group,
    }
    ci_info = ci_df.to_dict()

    return min_max_per_group, ci_info
