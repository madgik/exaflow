from __future__ import annotations

import numpy as np
from scipy.stats import norm

from exaflow.algorithms.federated.statistics.histogram import SimpleHistogram
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


class FederatedBinnedMannWhitneyUTest:
    """
    Federated histogram-based Mann-Whitney U test.

    Approximates ranks via histogram binning without sharing raw data across
    workers. Normal approximation follows scipy.stats.mannwhitneyu with
    method='asymptotic'.
    """

    def __init__(self, aggregator: NumpyAggregator):
        self.agg = aggregator
        self.hist = SimpleHistogram(aggregator)

    def compute(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alternative: str = "two-sided",
        use_continuity: bool = True,
        num_bins: int = 40,
    ) -> dict:
        """
        Perform Mann-Whitney U test on two independent samples.

        Args:
            x: First sample (group A values). Non-finite values (NaN, +/-inf)
                are dropped before the test.
            y: Second sample (group B values). Non-finite values (NaN, +/-inf)
                are dropped before the test.
            alternative: 'two-sided', 'less', or 'greater'.
            use_continuity: Apply continuity correction to z-score.
            num_bins: Number of histogram bins used for rank approximation.
                Must be at least 2.

        Returns:
            dict with u_stat, p_value, z_score, n1, n2.

            ``u_stat`` is always the U statistic of group A (``x``). ``z_score``
            is the standardized value of the U selected for the chosen
            ``alternative`` and so corresponds to the reported ``p_value`` tail,
            not necessarily to ``u_stat``: it uses ``u_stat`` for 'greater', the
            opposite U (``n1 * n2 - u_stat``) for 'less', and the larger of the
            two for 'two-sided'. For 'less', a large reported ``u_stat`` still
            yields the z-score of the opposite U.
        """
        if num_bins < 2:
            raise BadInputError("num_bins must be at least 2.")

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        n1 = int(self.agg.global_count(x))
        n2 = int(self.agg.global_count(y))

        if n1 == 0 or n2 == 0:
            raise BadInputError("Both groups must have at least one observation.")

        sum_ranks_y, bin_counts = self._rank(x, y, num_bins)

        u_stat = float(n1 * n2 + n2 * (n2 + 1) / 2 - sum_ranks_y)
        mu = n1 * n2 / 2.0
        n = n1 + n2

        tie_correction = sum(c**3 - c for c in bin_counts if c > 1)
        tie_correction = tie_correction / (n * (n - 1)) if n > 1 else 0.0

        sigma = np.sqrt(n1 * n2 * (n + 1 - tie_correction) / 12.0)
        if sigma == 0:
            raise BadInputError("Standard deviation is zero; cannot compute z-score.")

        # Mirror scipy.stats.mannwhitneyu(method="asymptotic"): pick the U for
        # the alternative, always test its upper tail, and apply the continuity
        # correction as a single -0.5 on that U (not "toward mu", which would
        # flip sign and disagree on opposite-direction one-sided alternatives).
        u2 = n1 * n2 - u_stat
        if alternative == "two-sided":
            u_for_p, factor = max(u_stat, u2), 2
        elif alternative == "greater":
            u_for_p, factor = u_stat, 1
        elif alternative == "less":
            u_for_p, factor = u2, 1
        else:
            raise BadInputError("alternative must be 'two-sided', 'less', or 'greater'")

        numerator = u_for_p - mu
        if use_continuity:
            numerator -= 0.5
        z = numerator / sigma
        p_value = float(min(factor * norm.sf(z), 1.0))

        return dict(
            u_stat=u_stat,
            p_value=p_value,
            z_score=float(z),
            n1=n1,
            n2=n2,
        )

    def _rank(
        self, x: np.ndarray, y: np.ndarray, num_bins: int
    ) -> tuple[float, np.ndarray]:
        combined = np.concatenate([x, y])
        overall_min = float(self.agg.global_min(combined))
        overall_max = float(self.agg.global_max(combined))
        if overall_min == overall_max:
            overall_max = overall_min + 1.0

        bounds = (overall_min, overall_max)
        counts_x, _ = self.hist.compute(x, num_bins, bounds=bounds)
        counts_y, _ = self.hist.compute(y, num_bins, bounds=bounds)
        total_counts = counts_x + counts_y

        # All values in a bin share the bin's average (mid) rank. The mid rank
        # is the count preceding the bin plus (bin_total + 1) / 2. Only the
        # rank sum of group y is needed downstream, so accumulate it directly
        # instead of materialising per-observation rank arrays.
        previous_counts = np.concatenate([[0.0], np.cumsum(total_counts)[:-1]])
        mid_ranks = previous_counts + (total_counts + 1) / 2.0
        sum_ranks_y = float(np.sum(counts_y * mid_ranks))

        return sum_ranks_y, total_counts
