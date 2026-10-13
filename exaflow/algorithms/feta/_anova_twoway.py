"""FETA-owned Type I/II ANOVA inference from completed model summaries."""

from typing import Dict
from typing import List
from typing import Optional

import scipy.stats as st

from exaflow.worker_communication import BadUserInput


def compute_anova_table(
    *,
    model_stats: Dict[str, Dict[str, float]],
    sstype: int,
    x1_name: str,
    x2_name: str,
) -> Dict[str, Dict[str, float]]:
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

    terms = [x1_name, x2_name, f"{x1_name}:{x2_name}", "Residuals"]

    if n_obs == 0 or rank_full == 0:
        return {
            "terms": terms,
            "sum_sq": {term: 0.0 for term in terms},
            "df": {term: 0 for term in terms},
            "f_stat": {term: None for term in terms},
            "f_pvalue": {term: None for term in terms},
        }

    if sstype == 1:
        ss_a = rss_const - rss_a
        ss_b = rss_a - rss_ab
    else:
        ss_a = rss_b - rss_ab
        ss_b = rss_a - rss_ab

    ss_inter = rss_ab - rss_full
    ss_resid = rss_full

    sum_sq_list = [ss_a, ss_b, ss_inter, ss_resid]
    sum_sq_list = [max(float(val), 0.0) for val in sum_sq_list]

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

    df_list = [df_a, df_b, df_inter, df_resid]

    ms = [0.0, 0.0, 0.0, 0.0]
    for idx in range(4):
        if df_list[idx] > 0:
            ms[idx] = sum_sq_list[idx] / df_list[idx]

    f_vals: List[Optional[float]] = [None, None, None, None]
    for idx in range(3):
        if df_list[idx] > 0 and df_resid > 0 and ms[3] != 0:
            f_vals[idx] = ms[idx] / ms[3]

    p_vals: List[Optional[float]] = [None, None, None, None]
    for idx in range(3):
        if f_vals[idx] is not None:
            p_vals[idx] = float(1.0 - st.f.cdf(f_vals[idx], df_list[idx], df_resid))

    return {
        "terms": terms,
        "sum_sq": dict(zip(terms, sum_sq_list)),
        "df": dict(zip(terms, df_list)),
        "f_stat": dict(zip(terms, f_vals)),
        "f_pvalue": dict(zip(terms, p_vals)),
    }
