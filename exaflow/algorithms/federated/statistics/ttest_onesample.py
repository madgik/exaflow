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
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client

    def compute(self, *, sample, mu: float, alpha: float, alternative: str):
        sample = to_numpy(sample).reshape(-1)
        n_obs = sample.size

        sum_x = sample.sum()
        sqrd_x = np.dot(sample, sample)
        diff_x = sum_x - n_obs * mu
        diff_sqrd_x = sqrd_x - 2 * mu * sum_x + n_obs * mu**2

        total_n_obs_arr = self.agg_client.sum(np.array([float(n_obs)], dtype=float))
        total_sum_x_arr = self.agg_client.sum(np.array([float(sum_x)], dtype=float))
        total_sqrd_x_arr = self.agg_client.sum(np.array([float(sqrd_x)], dtype=float))
        total_diff_x_arr = self.agg_client.sum(np.array([float(diff_x)], dtype=float))
        total_diff_sqrd_x_arr = self.agg_client.sum(
            np.array([float(diff_sqrd_x)], dtype=float)
        )
        total_n_obs = float(np.asarray(total_n_obs_arr).reshape(-1)[0])
        total_sum_x = float(np.asarray(total_sum_x_arr).reshape(-1)[0])
        total_sqrd_x = float(np.asarray(total_sqrd_x_arr).reshape(-1)[0])
        total_diff_x = float(np.asarray(total_diff_x_arr).reshape(-1)[0])
        total_diff_sqrd_x = float(np.asarray(total_diff_sqrd_x_arr).reshape(-1)[0])

        if total_n_obs <= 1:
            raise BadInputError("Not enough observations for one-sample t-test.")

        smpl_mean = total_sum_x / total_n_obs
        sd = np.sqrt(
            (total_diff_sqrd_x - (total_diff_x**2 / total_n_obs)) / (total_n_obs - 1)
        )
        sed = sd / np.sqrt(total_n_obs)
        t_stat = (smpl_mean - mu) / sed
        df = total_n_obs - 1

        ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=smpl_mean, scale=sed)

        if alternative == "greater":
            p_value = 1.0 - st.t.cdf(t_stat, df)
            ci_upper = "Infinity"
        elif alternative == "less":
            p_value = 1.0 - st.t.cdf(-t_stat, df)
            ci_lower = "-Infinity"
        else:
            p_value = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

        cohens_d = (smpl_mean - mu) / sd

        return dict(
            n_obs=int(total_n_obs),
            t_stat=t_stat,
            df=int(df),
            std=sd,
            p_value=p_value,
            mean_diff=smpl_mean,
            se_diff=sed,
            ci_upper=ci_upper,
            ci_lower=ci_lower,
            cohens_d=cohens_d,
        )
