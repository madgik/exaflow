from __future__ import annotations

import numpy as np
import scipy.stats as st

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import to_numpy


class FederatedTtestWelch:
    """
    Federated Welch's independent t-test using aggregated sufficient statistics.

    Welch's test does not assume equal population variances. It uses separate
    group variances in the standard error and Welch-Satterthwaite degrees of
    freedom.
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

        sample_a = _finite_sample(values[mask_a])
        sample_b = _finite_sample(values[mask_b])

        local_stats = np.array(
            [
                float(sample_a.size),
                float(sample_b.size),
                float(sample_a.sum()),
                float(sample_b.sum()),
                float(np.dot(sample_a, sample_a)),
                float(np.dot(sample_b, sample_b)),
            ],
            dtype=float,
        )
        (
            n_a,
            n_b,
            sum_a,
            sum_b,
            sq_sum_a,
            sq_sum_b,
        ) = np.asarray(self.agg_client.sum(local_stats), dtype=float).reshape(-1)

        if n_a < 2:
            raise BadInputError("Group A needs at least two finite observations.")
        if n_b < 2:
            raise BadInputError("Group B needs at least two finite observations.")

        mean_a = sum_a / n_a
        mean_b = sum_b / n_b
        var_a = _sample_variance(n=n_a, total=sum_a, sq_total=sq_sum_a, mean=mean_a)
        var_b = _sample_variance(n=n_b, total=sum_b, sq_total=sq_sum_b, mean=mean_b)

        var_term_a = var_a / n_a
        var_term_b = var_b / n_b
        se_diff = np.sqrt(var_term_a + var_term_b)
        if se_diff <= np.finfo(float).eps:
            raise BadInputError("Cannot compute Welch's t-test with zero variance.")

        mean_diff = mean_a - mean_b
        t_stat = mean_diff / se_diff
        df = _welch_degrees_of_freedom(
            var_term_a=var_term_a,
            var_term_b=var_term_b,
            n_a=n_a,
            n_b=n_b,
        )

        ci_lower, ci_upper = st.t.interval(
            1.0 - alpha, df, loc=mean_diff, scale=se_diff
        )
        p_value = _p_value(t_stat=t_stat, df=df, alternative=alternative)
        if alternative == "greater":
            ci_upper = "Infinity"
        elif alternative == "less":
            ci_lower = "-Infinity"

        effect_denominator = np.sqrt((var_a + var_b) / 2.0)
        if effect_denominator <= np.finfo(float).eps:
            raise BadInputError("Cannot compute Cohen's d with zero variance.")
        cohens_d = mean_diff / effect_denominator

        return {
            "t_stat": float(t_stat),
            "df": float(df),
            "p_value": float(p_value),
            "mean_diff": float(mean_diff),
            "se_diff": float(se_diff),
            "ci_upper": ci_upper,
            "ci_lower": ci_lower,
            "cohens_d": float(cohens_d),
        }


def _finite_sample(values):
    sample = to_numpy(values).reshape(-1)
    return sample[np.isfinite(sample)]


def _sample_variance(*, n: float, total: float, sq_total: float, mean: float) -> float:
    numerator = sq_total - 2.0 * mean * total + (mean**2) * n
    return float(max(numerator / (n - 1.0), 0.0))


def _welch_degrees_of_freedom(
    *,
    var_term_a: float,
    var_term_b: float,
    n_a: float,
    n_b: float,
) -> float:
    numerator = (var_term_a + var_term_b) ** 2
    denominator = (var_term_a**2 / (n_a - 1.0)) + (var_term_b**2 / (n_b - 1.0))
    if denominator <= np.finfo(float).eps:
        raise BadInputError("Cannot compute Welch degrees of freedom.")
    return float(numerator / denominator)


def _p_value(*, t_stat: float, df: float, alternative: str) -> float:
    if alternative == "greater":
        return float(st.t.sf(t_stat, df))
    if alternative == "less":
        return float(st.t.cdf(t_stat, df))
    if alternative == "two-sided":
        return float(st.t.sf(abs(t_stat), df) * 2.0)
    raise BadInputError(f"Unsupported alternative hypothesis: {alternative}")


def _squeeze_if_needed(series):
    if hasattr(series, "ndim") and series.ndim > 1:
        return series.squeeze()
    return series
