import numpy as np
from scipy.stats import norm

from library.descriptive_stats.histogram import FedHistogramSimple
from library.utils.interfaces.statistical_function import NumpyStatisticalFunction


class MannWhitneyUTest(NumpyStatisticalFunction):
    """
    Histogram-based implementation of the Mann-Whitney U test.

    Uses histogram binning to approximate ranks instead of exact ranking,
    which is more efficient for large datasets but less precise.
    """
    # The code is from deep-seek and follows the algorithm in scipy.stats


    def compute(
            self,
            x: np.ndarray,
            y: np.ndarray,
            alternative: str = 'two-sided',
            use_continuity: bool = True,
            num_bins: int = 10
    ):
        """
        Perform histogram-based Mann-Whitney U test on two independent samples.

        Bins the data into histograms and assigns average ranks to values within
        each bin, then computes the U statistic using normal approximation.

        Args:
            x: First sample of observations.
            y: Second sample of observations.
            alternative: 'two-sided', 'less', or 'greater'. Default is 'two-sided'.
            use_continuity: Apply continuity correction to z-score. Default is True.
            num_bins: Number of histogram bins for ranking. Default is 10.

        Returns:
            dict: Contains 'statistic', 'p_value', 'U1', 'U2', 'z_score',
                  'sigma', 'mu', and 'tie_correction'.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        n1 = self.aggregator.global_count(x)
        n2 = self.aggregator.global_count(y)

        # Rank the data
        ranks_x, ranks_y, counts = self.rank(x, y, num_bins)

        # Calculate U descriptive_stats using different formulas
        u_1 = np.sum(ranks_x) - n1 * (n1 + 1) / 2

        # Scipy's formula: U = n1*n2 + n2*(n2+1)/2 - sum(ranks_y)
        u_stat = n1 * n2 + n2 * (n2 + 1) / 2 - np.sum(ranks_y)

        # This should equal U1
        print(f"Verification: U1 = {u_1}, U_stat = {u_stat}")

        # Mean
        mu = n1 * n2 / 2.0
        n = n1 + n2

        # Tie correction
        tie_correction = 0
        for count in counts:
            if count > 1:
                tie_correction += (count ** 3 - count)
        tie_correction = tie_correction / (n * (n - 1)) if n > 1 else 0

        # Standard deviation with tie correction
        sigma = np.sqrt(n1 * n2 * (n + 1 - tie_correction) / 12.0)

        # Z-score with continuity correction (scipy's method)
        if use_continuity:
            # Correction towards the null hypothesis
            if u_stat > mu:
                z = (u_stat - mu - 0.5) / sigma
            else:
                z = (u_stat - mu + 0.5) / sigma
        else:
            z = (u_stat - mu) / sigma

        # P-value calculation using survival function
        if alternative == 'two-sided':
            p_value = 2 * norm.sf(np.abs(z))
        elif alternative == 'less':
            p_value = norm.cdf(z)
        elif alternative == 'greater':
            p_value = norm.sf(z)
        else:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

        return u_stat,p_value


    def rank(self,x, y, num_bins):
        # Rank the data
        overall_min = min([self.aggregator.global_min(x)[0], self.aggregator.global_min(y)[0]])
        overall_max = max([self.aggregator.global_max(x)[0], self.aggregator.global_max(y)[0]])
        # Federated Histogram
        histogram = FedHistogramSimple(self.aggregator)
        #
        counts_x,bin_edges_x =histogram.numerical_histogram(x, num_bins, value_range=(overall_min, overall_max))
        counts_y, bin_edges_y = histogram.numerical_histogram(y, num_bins, value_range=(overall_min, overall_max))
        total_counts = counts_x + counts_y
        # From histogrgam to ranks
        ranks_x = np.array([])
        ranks_y = np.array([])
        _previous_count = 0
        for _countx, _county, _hist_count in zip(counts_x, counts_y, total_counts):
            h_order = _previous_count + (_hist_count + 1) / 2
            ranks_x = np.hstack([ranks_x, np.full(int(_countx), h_order)])
            ranks_y = np.hstack([ranks_y, np.full(int(_county), h_order)])
            # ranks_x.extend([h_order] * _countx)
            # ranks_y.extend([h_order] * _county)
            _previous_count = _previous_count + _hist_count
        return ranks_x, ranks_y, total_counts








# # Test the implementation
# if __name__ == "__main__":
#     import random
#
#     group_a = np.asarray([random.randint(80, 140) for _ in range(100)])
#     group_b = np.asarray([random.randint(80, 140) for _ in range(100)])
#
#     print("Final implementation:")
#     u_stat,p_value = MannWhitneyUTest(None).compute(group_a, group_b, use_continuity=True, num_bins=100)
#     print(u_stat)
#     print(p_value)
#
#     print("\nScipy asymptotic method:")
#     from scipy.stats import mannwhitneyu
#
#     stat, p = mannwhitneyu(group_a, group_b, use_continuity=True, alternative='two-sided', method='asymptotic')
#     print(f"U statistic: {stat:.4f}")
#     print(f"P-value: {p:.6f}")