from __future__ import annotations

import numpy as np
import scipy.stats as st

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import to_numpy


class FederatedTTestIndependent:
    """
    Federated Student's independent t-test with aggregation-backed statistics.

    Uses pooled variance with df = n_a + n_b - 2. Confidence intervals follow
    the pooled-variance standard error. Cohen's d is computed using the pooled
    standard deviation.
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client

    def compute(
        self,
        *,
        data,
        group_var: str,
        value_var: str,
        group_a,
        group_b,
        alpha: float,
        alternative: str,
    ):
        grouping = _squeeze_if_needed(data[group_var])
        values = data[value_var]

        mask_a = _squeeze_if_needed(grouping == group_a)
        mask_b = _squeeze_if_needed(grouping == group_b)

        sample_a = to_numpy(values[mask_a]).reshape(-1)
        sample_b = to_numpy(values[mask_b]).reshape(-1)

        sample_a = to_numpy(sample_a).reshape(-1)
        sample_b = to_numpy(sample_b).reshape(-1)

        n_a = sample_a.size
        n_b = sample_b.size

        totals_arr = self.agg_client.sum(
            np.array(
                [
                    float(n_a),
                    float(n_b),
                    float(sample_a.sum()),
                    float(sample_b.sum()),
                ],
                dtype=float,
            )
        )
        sq_sums_arr = self.agg_client.sum(
            np.array(
                [
                    float(np.dot(sample_a, sample_a)),
                    float(np.dot(sample_b, sample_b)),
                ],
                dtype=float,
            )
        )
        n_a_total, n_b_total, sum_a_total, sum_b_total = np.asarray(
            totals_arr, dtype=float
        ).reshape(-1)
        sq_sum_a, sq_sum_b = np.asarray(sq_sums_arr, dtype=float).reshape(-1)

        if n_a_total < 1:
            raise BadInputError("Group A has no data.")
        if n_b_total < 1:
            raise BadInputError("Group B has no data.")
        if n_a_total + n_b_total <= 2:
            raise BadInputError("Not enough observations for independent t-test.")

        mean_a = sum_a_total / n_a_total
        mean_b = sum_b_total / n_b_total

        sd_a = np.sqrt(
            (sq_sum_a - 2 * mean_a * sum_a_total + (mean_a**2) * n_a_total)
            / (n_a_total - 1)
        )
        sd_b = np.sqrt(
            (sq_sum_b - 2 * mean_b * sum_b_total + (mean_b**2) * n_b_total)
            / (n_b_total - 1)
        )

        pooled_var = ((n_a_total - 1) * sd_a**2 + (n_b_total - 1) * sd_b**2) / (
            n_a_total + n_b_total - 2
        )
        sed = np.sqrt(pooled_var * (1 / n_a_total + 1 / n_b_total))
        t_stat = (mean_a - mean_b) / sed
        df = n_a_total + n_b_total - 2
        diff_mean = mean_a - mean_b

        ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=diff_mean, scale=sed)

        if alternative == "greater":
            p_value = 1.0 - st.t.cdf(t_stat, df)
            ci_upper = "Infinity"
        elif alternative == "less":
            p_value = 1.0 - st.t.cdf(-t_stat, df)
            ci_lower = "-Infinity"
        else:
            p_value = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

        cohens_d = diff_mean / np.sqrt(pooled_var)

        return dict(
            t_stat=t_stat,
            df=float(df),
            p_value=p_value,
            mean_diff=diff_mean,
            se_diff=sed,
            ci_upper=ci_upper,
            ci_lower=ci_lower,
            cohens_d=cohens_d,
        )


def _squeeze_if_needed(series):
    if hasattr(series, "ndim") and series.ndim > 1:
        return series.squeeze()
    return series
