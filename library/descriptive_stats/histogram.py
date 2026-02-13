from library.utils.interfaces.statistical_function import  NumpyStatisticalFunction
import numpy as np

class FedHistogramSimple(NumpyStatisticalFunction):

    def numerical_histogram(self, x:np.ndarray, num_bins:int, *, value_range=None):
        if x.ndim != 1:
            raise ValueError(f"Input must be 1-dimensional, got {x.ndim} dimensions")
        if value_range is None:
            min_val = self.aggregator.global_min(x)
            max_val = self.aggregator.global_max(x)
            counts, bin_edges = np.histogram(x, bins=num_bins, range=(min_val,max_val))
        else:
            counts, bin_edges = np.histogram(x, bins=num_bins, range=value_range)
        # StandardHistogram.plot_histogram(counts, bin_edges)
        counts = self.aggregator.fed_sum(counts)
        # StandardHistogram.plot_histogram(counts, bin_edges)
        return counts, bin_edges
    #
    # def categorical_histogram(self, x, categories=None):
    #     if x.ndim != 1:
    #         raise ValueError(f"Input must be 1-dimensional, got {x.ndim} dimensions")
    #
    #     # Get unique categories and their counts
    #     if categories is None:
    #         unique_cats, counts = np.unique(x, return_counts=True)
    #     else:
    #         # Use specified categories
    #         unique_cats = np.asarray(categories)
    #         counts = np.array([np.sum(x == cat) for cat in categories])
    #
    #     # Apply federated sum (like your numerical version)
    #     counts = self.aggregator.fed_sum(counts)
    #
    #     # Optional: Plot if you have plotting functionality
    #     # self.plot_categorical_histogram(counts, unique_cats)
    #
    #     return counts, unique_cats
