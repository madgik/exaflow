from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import special

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import to_numpy


class PearsonCorrelationResult:
    def __init__(
        self,
        *,
        n_obs: int,
        correlations,
        p_values,
        ci_hi,
        ci_lo,
    ):
        self.n_obs = n_obs
        self.correlations = correlations
        self.p_values = p_values
        self.ci_hi = ci_hi
        self.ci_lo = ci_lo


class FederatedPearsonCorrelation:
    """
    Statsmodels-style Pearson correlation backed by federated aggregation.

    ``corrcoef`` returns correlation, p-values, and confidence intervals for
    each (x, y) pair using aggregation primitives and the specified ``alpha``.
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client

    def corrcoef(
        self,
        *,
        data: pd.DataFrame,
        x_vars: List[str],
        y_vars: List[str],
        alpha: float,
    ) -> PearsonCorrelationResult:
        x = to_numpy(data[x_vars])
        y = to_numpy(data[y_vars])
        n_obs = len(y)

        total_n_obs_arr = self.agg_client.sum(np.array([float(n_obs)], dtype=float))
        total_sx = self.agg_client.sum(np.einsum("ij->j", x))
        total_sy = self.agg_client.sum(np.einsum("ij->j", y))
        total_n_obs = float(np.asarray(total_n_obs_arr).reshape(-1)[0])

        df = total_n_obs - 2
        if total_n_obs == 0:
            raise BadInputError("Cannot compute Pearson correlation on empty data.")

        if df <= 0:
            raise BadInputError(
                "Not enough observations to compute Pearson correlation."
            )

        x_centered = x - np.asarray(total_sx, dtype=float) / total_n_obs
        y_centered = y - np.asarray(total_sy, dtype=float) / total_n_obs
        total_sxx = self.agg_client.sum(np.einsum("ij,ij->j", x_centered, x_centered))
        total_syy = self.agg_client.sum(np.einsum("ij,ij->j", y_centered, y_centered))
        total_sxy = self.agg_client.sum(np.einsum("ji,jk->ki", x_centered, y_centered))
        total_sxx = np.asarray(total_sxx, dtype=float)
        total_syy = np.asarray(total_syy, dtype=float)
        total_sxy = np.asarray(total_sxy, dtype=float)

        d = np.sqrt(total_sxx) * np.sqrt(total_syy)[:, np.newaxis]
        correlations = total_sxy / d
        correlations[d == 0] = 0
        correlations = correlations.clip(-1, 1)
        t_squared = correlations**2 * (
            df / ((1.0 - correlations) * (1.0 + correlations))
        )
        p_values = special.betainc(
            0.5 * df, 0.5, np.fmin(np.asarray(df / (df + t_squared)), 1.0)
        )
        p_values[abs(correlations) == 1] = 0
        r_z = np.arctanh(correlations)
        se = 1 / np.sqrt(total_n_obs - 3)
        z = st.norm.ppf(1 - alpha / 2)
        lo_z, hi_z = r_z - z * se, r_z + z * se
        ci_lo, ci_hi = np.tanh((lo_z, hi_z))

        return PearsonCorrelationResult(
            n_obs=int(total_n_obs),
            correlations=correlations.tolist(),
            p_values=p_values.tolist(),
            ci_lo=ci_lo.tolist(),
            ci_hi=ci_hi.tolist(),
        )
