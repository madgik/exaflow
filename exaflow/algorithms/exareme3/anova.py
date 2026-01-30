from typing import Dict
from typing import List
from typing import Optional

import scipy.stats as st
from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.ols import FederatedOLS
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "anova"


class AnovaResult(BaseModel):
    terms: List[str]
    sum_sq: List[float]
    df: List[int]
    f_stat: List[Optional[float]]
    f_pvalue: List[Optional[float]]


class AnovaTwoWayAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self):
        y = self.inputdata.y[0]
        xs = self.inputdata.x
        if len(xs) != 2:
            raise BadUserInput("ANOVA two-way requires exactly two covariates (x).")
        x1, x2 = xs

        sstype = self.get_parameter("sstype")

        levels_a = list(self.metadata[x1]["enumerations"])
        levels_b = list(self.metadata[x2]["enumerations"])
        if len(levels_a) < 2:
            raise BadUserInput(
                f"The variable {x1} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )
        if len(levels_b) < 2:
            raise BadUserInput(
                f"The variable {x2} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )

        results = self.run_local_udf(
            func=anova_twoway_local_step,
            kw_args={
                "x1": x1,
                "x2": x2,
                "y": y,
                "levels_a": levels_a,
                "levels_b": levels_b,
            },
        )
        models = results[0]
        metrics = compute_anova_from_models(
            model_stats=models,
            sstype=sstype,
            x1_name=x1,
            x2_name=x2,
        )
        return AnovaResult(**metrics)


@exareme3_udf(with_aggregation_server=True)
def anova_twoway_local_step(agg_client, data, x1, x2, y, levels_a, levels_b):
    import numpy as np
    import pandas as pd
    from patsy import dmatrix

    def _build_dataframe():
        subset = data[[y, x1, x2]].copy()
        subset.dropna(inplace=True)
        subset[x1] = pd.Categorical(subset[x1], categories=levels_a)
        subset[x2] = pd.Categorical(subset[x2], categories=levels_b)
        subset[y] = subset[y].astype(float)
        return subset

    df = _build_dataframe()

    # Explicit levels ensure consistent design matrices across workers
    levels_a_repr = repr(list(levels_a))
    levels_b_repr = repr(list(levels_b))
    formulas = {
        "const": "1",
        "a": f"C({x1}, levels={levels_a_repr})",
        "b": f"C({x2}, levels={levels_b_repr})",
        "ab": f"C({x1}, levels={levels_a_repr}) + C({x2}, levels={levels_b_repr})",
        "full": f"C({x1}, levels={levels_a_repr}) * C({x2}, levels={levels_b_repr})",
    }

    design_frames = {
        name: dmatrix(formula, df, return_type="dataframe")
        for name, formula in formulas.items()
    }
    ref_index = next(iter(design_frames.values())).index
    y_vector = df.loc[ref_index, y].to_numpy(dtype=float).reshape(-1, 1)
    design_mats = {
        name: np.asarray(mat, dtype=float) for name, mat in design_frames.items()
    }

    def _fit_model(X: np.ndarray):
        model = FederatedOLS(agg_client=agg_client)
        model.fit(X, y_vector)
        return {
            "rss": model.rss,
            "n_obs": model.nobs,
            "rank": model.rank_,
        }

    return {name: _fit_model(X) for name, X in design_mats.items()}


def compute_anova_from_models(
    *,
    model_stats: Dict[str, Dict[str, float]],
    sstype: int,
    x1_name: str,
    x2_name: str,
) -> Dict[str, List]:
    def _get(key: str):
        return (
            model_stats[key]["rss"],
            model_stats[key]["rank"],
            model_stats[key]["n_obs"],
        )

    rss_const, rank_const, n_obs = _get("const")
    rss_a, rank_a, _ = _get("a")
    rss_b, rank_b, _ = _get("b")
    rss_ab, rank_ab, _ = _get("ab")
    rss_full, rank_full, _ = _get("full")

    if n_obs == 0 or rank_full == 0:
        return {
            "terms": [x1_name, x2_name, f"{x1_name}:{x2_name}", "Residuals"],
            "sum_sq": [0.0, 0.0, 0.0, 0.0],
            "df": [0, 0, 0, 0],
            "f_stat": [None, None, None, None],
            "f_pvalue": [None, None, None, None],
        }

    if sstype == 1:
        ss_a = rss_const - rss_a
        ss_b = rss_a - rss_ab
    else:
        ss_a = rss_b - rss_ab
        ss_b = rss_a - rss_ab

    ss_inter = rss_ab - rss_full
    ss_resid = rss_full

    sum_sq = [ss_a, ss_b, ss_inter, ss_resid]
    sum_sq = [max(float(val), 0.0) for val in sum_sq]

    df_a = max(rank_a - rank_const, 0)
    df_b = max(rank_b - rank_const, 0)
    df_inter = max(rank_full - rank_ab, 0)
    df_resid = max(n_obs - rank_full, 0)

    if df_a == 0:
        raise BadUserInput(
            f"The data of variable {x1_name} contain less than 2 levels and "
            "Anova cannot be performed. Please select more data or choose another "
            "variable."
        )
    if df_b == 0:
        raise BadUserInput(
            f"The data of variable {x2_name} contain less than 2 levels and "
            "Anova cannot be performed. Please select more data or choose another "
            "variable."
        )

    df = [df_a, df_b, df_inter, df_resid]

    ms = [0.0, 0.0, 0.0, 0.0]
    for idx in range(4):
        if df[idx] > 0:
            ms[idx] = sum_sq[idx] / df[idx]

    f_vals: List[Optional[float]] = [None, None, None, None]
    for idx in range(3):
        if df[idx] > 0 and df_resid > 0 and ms[3] != 0:
            f_vals[idx] = ms[idx] / ms[3]

    p_vals: List[Optional[float]] = [None, None, None, None]
    for idx in range(3):
        if f_vals[idx] is not None:
            p_vals[idx] = float(1.0 - st.f.cdf(f_vals[idx], df[idx], df_resid))

    terms = [x1_name, x2_name, f"{x1_name}:{x2_name}", "Residuals"]

    return {
        "terms": terms,
        "sum_sq": sum_sq,
        "df": df,
        "f_stat": f_vals,
        "f_pvalue": p_vals,
    }
