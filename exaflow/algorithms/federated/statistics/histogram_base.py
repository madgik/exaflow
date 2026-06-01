import math
from typing import Tuple

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


def _wilkinson_step(
    data_min: float, data_max: float, num_bins: int
) -> Tuple[float, float, float, int]:
    """Wilkinson nice-numbers bin sizing.

    Snaps the bin width to the nearest 1/2/5 × 10^n, then extends min/max
    outward to the nearest step boundary.

    Returns:
        (step, new_min, new_max, new_num_bins)
    """
    raw_step = (data_max - data_min) / num_bins
    if raw_step == 0:
        return 1.0, data_min - 0.5, data_min + 0.5, 1
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = min([1, 2, 5], key=lambda m: abs(raw_step - m * magnitude)) * magnitude
    new_min = math.floor(data_min / step) * step
    new_max = math.ceil(data_max / step) * step
    new_num_bins = round((new_max - new_min) / step)
    return step, new_min, new_max, new_num_bins


class SimpleHistogram:
    """A simplified federated histogram algorithm for numerical data.

    This class provides a straightforward implementation of a federated histogram
    by first determining global bounds and then aggregating local counts.
    """

    def __init__(self, aggregator: NumpyAggregator):
        """Initialize the SimpleHistogram with a NumpyAggregator.

        Args:
            aggregator: An instance of NumpyAggregator used for federated
                       operations like global min/max and sum.
        """
        self.agg = aggregator

    def compute(
        self, x: np.ndarray, num_bins: int, *, bounds: Tuple[float, float] | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the federated histogram of a 1D array across all clients.

        If bounds is not provided, the global minimum and maximum are automatically
        computed and used as the binning range.

        Args:
            x: Local data array for this client.
            num_bins: The number of equal-width bins in the histogram.
            bounds: The lower and upper range of the bins. If None, it is
                  automatically determined from the global data.

        Returns:
            A tuple containing:
            - counts: Federated bin counts (sum of local counts across all clients).
            - bin_edges: The edges of the bins used for the histogram.

        Raises:
            BadInputError: If 'x' is not a 1D array or if no finite data is
                          available globally to determine histogram bounds.
        """
        if x.ndim != 1:
            raise BadInputError(f"Input 'x' must be a 1D array, but got {x.ndim}D.")

        if bounds is None:
            min_val = float(self.agg.global_min(x))
            max_val = float(self.agg.global_max(x))
            if not np.isfinite(min_val) or not np.isfinite(max_val):
                raise BadInputError("No finite data available to compute histogram.")
            if min_val == max_val:
                max_val = min_val + 1.0

        x_clean = x[~np.isnan(x)]
        range_ = bounds if bounds is not None else (min_val, max_val)
        counts, bin_edges = np.histogram(x_clean, bins=num_bins, range=range_)

        counts = self.agg.fed_sum(counts)
        return counts, bin_edges


class CategoricalHistogram:
    """A federated histogram algorithm for categorical (string) data.

    This class computes the frequency of each unique category across all clients.
    """

    def __init__(self, aggregator: NumpyAggregator):
        """Initialize the CategoricalHistogram with a NumpyAggregator.

        Args:
            aggregator: An instance of NumpyAggregator used for federated
                       operations like global union and sum.
        """
        self.agg = aggregator

    def compute(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the federated histogram of a categorical array.

        Args:
            x: Local data array for this client (typically strings).

        Returns:
            A tuple containing:
            - categories: A numpy array of unique global categories (dtype=object).
            - counts: Federated counts for each category (dtype=float).

        Raises:
            BadInputError: If 'x' is not a 1D array.
        """
        if x.ndim != 1:
            raise BadInputError(f"Input 'x' must be a 1D array, but got {x.ndim}D.")

        # Ensure we work with object array to handle None/NaN consistently
        x_obj = np.asarray(x, dtype=object)
        mask = ~pd.isna(x_obj)

        if not np.any(mask):
            # No data case
            global_categories_raw = self.agg.fed_union(np.array([], dtype=object))
            global_categories = np.array(
                sorted([str(cat) for cat in global_categories_raw]), dtype=object
            )
            if len(global_categories) == 0:
                return np.array([], dtype=object), np.array([], dtype=float)
            local_counts_dict = {}
        else:
            # Get local unique elements and their counts in one go
            # Stringify for sorting homogeneity
            clean_x = x_obj[mask].astype(str)
            unique_elements, counts_elements = np.unique(clean_x, return_counts=True)
            local_counts_dict = dict(zip(unique_elements, counts_elements))

            # Determine global set of categories using the unique elements found
            global_categories_raw = self.agg.fed_union(unique_elements)
            global_categories = np.array(
                sorted([str(cat) for cat in global_categories_raw]), dtype=object
            )

        local_counts = np.array(
            [float(local_counts_dict.get(cat, 0)) for cat in global_categories],
            dtype=int,
        )

        # Aggregate counts globally
        global_counts = self.agg.fed_sum(local_counts)

        return global_categories, global_counts


class WilkinsonHistogram:
    """Federated histogram for numerical data using Wilkinson nice-number bin sizing.

    Derives bin width by snapping to the nearest 1/2/5 × 10^n value, then extends
    the range outward to clean boundaries. The actual number of bins may therefore
    differ from the requested hint.
    """

    def __init__(self, aggregator: NumpyAggregator):
        self.agg = aggregator

    def compute(self, x: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a federated histogram with Wilkinson-snapped bin boundaries.

        Args:
            x: Local 1D data array for this client.
            num_bins: Hint for the desired number of bins; actual count may differ
                after snapping to a nice step size.

        Returns:
            (counts, bin_edges) where bin_edges are aligned to the nice step.

        Raises:
            BadInputError: If x is not 1D or if no finite data is available.
        """
        if x.ndim != 1:
            raise BadInputError(f"Input 'x' must be a 1D array, but got {x.ndim}D.")

        min_val = self.agg.global_min(x)
        max_val = self.agg.global_max(x)
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            raise BadInputError("No finite data available to compute histogram.")

        _, new_min, new_max, new_num_bins = _wilkinson_step(
            float(min_val), float(max_val), num_bins
        )

        x_clean = x[~np.isnan(x)]
        counts, bin_edges = np.histogram(
            x_clean, bins=new_num_bins, range=(new_min, new_max)
        )
        counts = self.agg.fed_sum(counts.astype(float))
        return counts, bin_edges
