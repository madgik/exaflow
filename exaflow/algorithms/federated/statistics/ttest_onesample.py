from __future__ import annotations

import numpy as np
import scipy.stats as st

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import to_numpy


class FederatedTTestOneSample:
    """
    Federated one-sample t-test with aggregation-backed statistics.

    Tests the mean of a numeric sample against a hypothesized mean ``mu``.
    Uses df = n - 1 and standard errors from the sample standard deviation.
    ``mean_diff`` and its confidence interval refer to ``mean - mu``.
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client

    def compute(self, *, sample, mu: float, alpha: float, alternative: str):
        sample = to_numpy(sample).reshape(-1)

        totals_arr = self.agg_client.sum(
            np.array([float(sample.size), float(sample.sum())], dtype=float)
        )
        total_n_obs, total_sum_x = np.asarray(totals_arr, dtype=float).reshape(-1)

        if total_n_obs <= 1:
            raise BadInputError("Not enough observations for one-sample t-test.")

        smpl_mean = total_sum_x / total_n_obs
        centered = sample - smpl_mean
        total_ssd_arr = self.agg_client.sum(
            np.array([float(np.dot(centered, centered))], dtype=float)
        )
        total_ssd = float(np.asarray(total_ssd_arr).reshape(-1)[0])

        sd = np.sqrt(total_ssd / (total_n_obs - 1))
        sed = sd / np.sqrt(total_n_obs)
        mean_diff = smpl_mean - mu
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

        cohens_d = mean_diff / sd

        return dict(
            n_obs=int(total_n_obs),
            t_stat=t_stat,
            df=int(df),
            std=sd,
            p_value=p_value,
            mean_diff=mean_diff,
            se_diff=sed,
            ci_upper=ci_upper,
            ci_lower=ci_lower,
            cohens_d=cohens_d,
        )
