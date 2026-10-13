from __future__ import annotations

import numpy as np
import scipy.stats as st

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import to_numpy


class FederatedTTestPaired:
    """
    Federated paired t-test with aggregation-backed statistics.

    Computes the t-test on paired differences with df = n - 1. Cohen's d is
    based on mean(diff) / std(diff).
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client

    def compute(self, *, sample_x, sample_y, alpha: float, alternative: str):
        sample_x = to_numpy(sample_x).reshape(-1)
        sample_y = to_numpy(sample_y).reshape(-1)
        if sample_x.shape != sample_y.shape:
            raise BadInputError("Paired samples must have the same length.")

        diff = sample_x - sample_y

        totals_arr = self.agg_client.sum(
            np.array([float(diff.size), float(diff.sum())], dtype=float)
        )
        total_n_obs, total_diff_sum = np.asarray(totals_arr, dtype=float).reshape(-1)

        if total_n_obs <= 1:
            raise BadInputError("Not enough observations for paired t-test.")

        mean_diff = total_diff_sum / total_n_obs
        centered = diff - mean_diff
        total_ssd_arr = self.agg_client.sum(
            np.array([float(np.dot(centered, centered))], dtype=float)
        )
        total_ssd = float(np.asarray(total_ssd_arr).reshape(-1)[0])

        sd_diff = np.sqrt(total_ssd / (total_n_obs - 1))
        sed = sd_diff / np.sqrt(total_n_obs)
        t_stat = mean_diff / sed
        df = total_n_obs - 1

        ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=mean_diff, scale=sed)

        if alternative == "greater":
            p_value = 1.0 - st.t.cdf(t_stat, df)
            ci_upper = "Infinity"
        elif alternative == "less":
            p_value = 1.0 - st.t.cdf(-t_stat, df)
            ci_lower = "-Infinity"
        else:
            p_value = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

        cohens_d = mean_diff / sd_diff

        return dict(
            t_stat=t_stat,
            df=int(df),
            p_value=p_value,
            mean_diff=mean_diff,
            se_diff=sed,
            ci_upper=ci_upper,
            ci_lower=ci_lower,
            cohens_d=cohens_d,
        )
