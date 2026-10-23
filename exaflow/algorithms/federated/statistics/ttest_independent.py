from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.statistics.t_test_inference import (
    t_test_p_value_and_ci,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import to_numpy
from exaflow.algorithms.federated.validation import check_choice
from exaflow.algorithms.federated.validation import check_dataframe
from exaflow.algorithms.federated.validation import check_scalar


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
        check_scalar(alpha, name="alpha", minimum=0, maximum=1)
        check_choice(alternative, ("two-sided", "less", "greater"), name="alternative")
        check_dataframe(data, (group_var, value_var))
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

        # Pool sums of squares directly: a group of size 1 contributes 0 instead
        # of 0 * NaN through its undefined sample variance.
        ss_a = sq_sum_a - 2 * mean_a * sum_a_total + (mean_a**2) * n_a_total
        ss_b = sq_sum_b - 2 * mean_b * sum_b_total + (mean_b**2) * n_b_total
        pooled_var = (ss_a + ss_b) / (n_a_total + n_b_total - 2)
        sed = np.sqrt(pooled_var * (1 / n_a_total + 1 / n_b_total))
        t_stat = (mean_a - mean_b) / sed
        df = n_a_total + n_b_total - 2
        diff_mean = mean_a - mean_b

        p_value, ci_lower, ci_upper = t_test_p_value_and_ci(
            t_stat=t_stat,
            df=df,
            estimate=diff_mean,
            se=sed,
            alpha=alpha,
            alternative=alternative,
        )

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
