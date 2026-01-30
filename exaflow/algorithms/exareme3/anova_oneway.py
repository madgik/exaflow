from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.anova_oneway import FederatedAnovaOneWay
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "anova_oneway"


class AnovaResult(BaseModel):
    anova_table: Dict
    tuckey_test: List[Dict]
    min_max_per_group: Dict
    ci_info: Dict


class AnovaOneWayAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        """
        Exaflow implementation of one-way ANOVA with Tukey HSD, matching the
        behavior of the original exaflow ANOVA_ONEWAY algorithm.
        """
        y_var_name = self.inputdata.y[0]
        x_var_name = self.inputdata.x[0]
        covar_enums = self.metadata[x_var_name].get("enumerations")

        # Run a single distributed ANOVA UDF
        udf_results = self.run_local_udf(
            func=anova_oneway_local_step,
            kw_args={
                "x_var": x_var_name,
                "y_var": y_var_name,
                "covar_enums": covar_enums,
            },
        )

        result = udf_results[0]

        anova_result = {
            "n_obs": result["n_obs"],
            "y_label": y_var_name,
            "x_label": x_var_name,
            "df_residual": result["df_residual"],
            "df_explained": result["df_explained"],
            "ss_residual": result["ss_residual"],
            "ss_explained": result["ss_explained"],
            "ms_residual": result["ms_residual"],
            "ms_explained": result["ms_explained"],
            "p_value": result["p_value"],
            "f_stat": result["f_stat"],
        }

        # Tukey table into list-of-dicts (records)
        tukey_result = {
            "groupA": result["thsd A"],
            "groupB": result["thsd B"],
            "meanA": result["thsd mean(A)"],
            "meanB": result["thsd mean(B)"],
            "diff": result["thsd diff"],
            "se": result["thsd Std.Err."],
            "t_stat": result["thsd t value"],
            "p_tuckey": result["thsd Pr(>|t|)"],
        }
        df_tukey_result = pd.DataFrame(tukey_result)
        tukey_test = df_tukey_result.to_dict(orient="records")

        means = result["means"]
        sample_stds = result["sample_stds"]
        categories = result["categories"]
        var_min_per_group = result["var_min_per_group"]
        var_max_per_group = result["var_max_per_group"]
        group_stats_index = result["group_stats_index"]

        min_max_per_group, ci_info = get_min_max_ci_info(
            means=means,
            sample_stds=sample_stds,
            categories=categories,
            var_min_per_group=var_min_per_group,
            var_max_per_group=var_max_per_group,
            group_stats_index=group_stats_index,
        )

        return AnovaResult(
            anova_table=anova_result,
            tuckey_test=tukey_test,
            min_max_per_group=min_max_per_group,
            ci_info=ci_info,
        )


@exareme3_udf(with_aggregation_server=True)
def anova_oneway_local_step(agg_client, data, x_var, y_var, covar_enums):
    covar_enums = list(covar_enums)

    # --- Local stats like original local1, but force 1D arrays ---
    y_col = data[y_var]
    x_col = data[x_var]

    # If for any reason we got a DataFrame, fall back to the first column
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    if isinstance(x_col, pd.DataFrame):
        x_col = x_col.iloc[:, 0]

    y_col = y_col.reset_index(drop=True)
    x_col = x_col.reset_index(drop=True)

    if len(covar_enums) < 2:
        raise BadUserInput(
            "Cannot perform Anova one-way. Covariable has only one level."
        )

    groups = []
    for category in covar_enums:
        mask = x_col == category
        groups.append(y_col[mask].to_numpy(dtype=float, copy=False))

    model = FederatedAnovaOneWay(agg_client=agg_client)
    try:
        model.fit(groups=groups, categories=covar_enums)
    except ValueError as exc:
        raise BadUserInput(str(exc))

    return {
        "n_obs": int(model.nobs),
        "categories": model.categories_,
        "means": model.means_,
        "sample_stds": model.sample_stds_,
        "var_min_per_group": model.var_min_per_group_,
        "var_max_per_group": model.var_max_per_group_,
        "group_stats_index": model.group_stats_index_,
        "df_explained": float(model.df_between),
        "df_residual": float(model.df_within),
        "ss_explained": float(model.ss_between),
        "ss_residual": float(model.ss_within),
        "ms_explained": float(model.ms_between),
        "ms_residual": float(model.ms_within),
        "f_stat": float(model.fvalue),
        "p_value": float(model.pvalue),
        "thsd A": model.thsd_["A"].tolist(),
        "thsd B": model.thsd_["B"].tolist(),
        "thsd mean(A)": model.thsd_["mean(A)"].tolist(),
        "thsd mean(B)": model.thsd_["mean(B)"].tolist(),
        "thsd diff": model.thsd_["diff"].tolist(),
        "thsd Std.Err.": model.thsd_["Std.Err."].tolist(),
        "thsd t value": model.thsd_["t value"].tolist(),
        "thsd Pr(>|t|)": model.thsd_["Pr(>|t|)"].tolist(),
    }


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
