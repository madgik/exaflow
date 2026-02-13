import numpy as np

from library.descriptive_stats.histogram import FedHistogramSimple
from library.utils.interfaces.statistical_function import NumpyStatisticalFunction
import math

class MedianBasedOnHistogram(NumpyStatisticalFunction):

    def compute(self, x: np.ndarray, *, range_acc=0.1):
        # Check that x is 1D
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError(
                f"Input x must be a 1D numpy array, got shape {x.shape if hasattr(x, 'shape') else 'unknown'}")
        # Check that range_acc is >= 0.01
        if range_acc < 0.01:
            raise ValueError(f"range_acc must be at least 0.01, got {range_acc}")

        bin_count =math.ceil(1/range_acc)
        hist = FedHistogramSimple(self.aggregator)
        counts, bin_edges = hist.numerical_histogram(x, bin_count)
        return MedianBasedOnHistogram._compute_median_from_histogram(counts, bin_edges)

    @staticmethod
    def _compute_median_from_histogram(counts, bin_edges):
        """
        Compute the median from a histogram given bin_edges and counts.

        :param bin_edges: List of bin edges. Length = num_bins + 1.
        :param counts: List of bin counts. Length = num_bins.
        :return: Median value.
        """
        total_count = sum(counts)
        half_total = total_count / 2

        cumulative = 0
        for i, count in enumerate(counts):
            cumulative += count
            if cumulative >= half_total:
                bin_start = bin_edges[i]
                bin_end = bin_edges[i + 1]
                bin_width = bin_end - bin_start

                cumulative_before_bin = cumulative - count
                median_offset = (half_total - cumulative_before_bin) / count

                median = bin_start + median_offset * bin_width
                return median
        return None
