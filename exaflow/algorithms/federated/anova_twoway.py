from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as st
from patsy import dmatrix

from exaflow.algorithms.federated.ols import FederatedOLS


class FederatedAnovaTwoWay:
    """
    Federated two-way ANOVA with a statsmodels-like interface.

    This estimator mirrors statsmodels' Type-II ANOVA for the model
    y ~ A * B using distributed sufficient statistics.
    """

    def __init__(self, agg_client, *, sstype: int = 2):
        """
        Initialize a federated two-way ANOVA estimator.

        Parameters
        ----------
        agg_client
            Aggregation client used to sum local sufficient statistics.
        sstype
            Sum-of-squares type (1 or 2). Defaults to Type II.
        """
        self.agg_client = agg_client
        self.sstype = sstype

    def fit(
        self,
        *,
        data: pd.DataFrame,
        y: str,
        x1: str,
        x2: str,
        levels_a: List,
        levels_b: List,
        sstype: Optional[int] = None,
    ):
        """
        Fit two-way ANOVA using federated OLS models.

        Parameters
        ----------
        data
            Local pandas DataFrame containing y, x1, x2 columns.
        y
            Response variable name.
        x1
            First categorical factor name.
        x2
            Second categorical factor name.
        levels_a
            Enumerations for x1.
        levels_b
            Enumerations for x2.
        sstype
            Optional override for sum-of-squares type.
        """
        levels_a = list(levels_a)
        levels_b = list(levels_b)
        if len(levels_a) < 2:
            raise ValueError(
                f"The variable {x1} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )
        if len(levels_b) < 2:
            raise ValueError(
                f"The variable {x2} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )

        df = self._build_dataframe(
            data=data, y=y, x1=x1, x2=x2, levels_a=levels_a, levels_b=levels_b
        )
        model_stats = self._fit_models(
            df=df, y=y, x1=x1, x2=x2, levels_a=levels_a, levels_b=levels_b
        )
        sstype = self.sstype if sstype is None else sstype

        results = self._compute_anova_table(
            model_stats=model_stats,
            sstype=sstype,
            x1_name=x1,
            x2_name=x2,
        )

        self.terms_ = results["terms"]
        self.sum_sq_ = results["sum_sq"]
        self.df_ = results["df"]
        self.f_stat_ = results["f_stat"]
        self.f_pvalue_ = results["f_pvalue"]
        return self

    def _build_dataframe(
        self,
        *,
        data: pd.DataFrame,
        y: str,
        x1: str,
        x2: str,
        levels_a: List,
        levels_b: List,
    ) -> pd.DataFrame:
        subset = data[[y, x1, x2]].copy()
        subset.dropna(inplace=True)
        subset[x1] = pd.Categorical(subset[x1], categories=levels_a)
        subset[x2] = pd.Categorical(subset[x2], categories=levels_b)
        subset[y] = subset[y].astype(float)
        return subset

    def _fit_models(
        self,
        *,
        df: pd.DataFrame,
        y: str,
        x1: str,
        x2: str,
        levels_a: List,
        levels_b: List,
    ) -> Dict[str, Dict[str, float]]:
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
            model = FederatedOLS(agg_client=self.agg_client)
            model.fit(X, y_vector)
            return {
                "rss": float(model.rss),
                "n_obs": int(model.nobs),
                "rank": int(model.rank_),
            }

        return {name: _fit_model(X) for name, X in design_mats.items()}

    def _compute_anova_table(
        self,
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
            raise ValueError(
                f"The data of variable {x1_name} contain less than 2 levels and "
                "Anova cannot be performed. Please select more data or choose another "
                "variable."
            )
        if df_b == 0:
            raise ValueError(
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
