from __future__ import annotations

import scipy.stats as st


def t_test_p_value_and_ci(*, t_stat, df, estimate, se, alpha: float, alternative: str):
    """P-value and (1 - alpha) confidence interval for a t-test estimate.

    One-sided alternatives use the one-sided critical value t_{1-alpha} and an
    unbounded side, matching scipy.stats and R's t.test. Tail probabilities use
    the survival function so small p-values do not underflow to 0.
    """
    if alternative == "greater":
        ci_lower = estimate - st.t.ppf(1.0 - alpha, df) * se
        return st.t.sf(t_stat, df), ci_lower, "Infinity"
    if alternative == "less":
        ci_upper = estimate + st.t.ppf(1.0 - alpha, df) * se
        return st.t.cdf(t_stat, df), "-Infinity", ci_upper
    ci_lower, ci_upper = st.t.interval(1.0 - alpha, df, loc=estimate, scale=se)
    return 2.0 * st.t.sf(abs(t_stat), df), ci_lower, ci_upper
