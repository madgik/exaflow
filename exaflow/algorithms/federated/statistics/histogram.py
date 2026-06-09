from __future__ import annotations

import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)

HistogramBin = Union[float, str]


def _wilkinson_step(
    data_min: float, data_max: float, num_bins: int, *, min_step: float = 0.0
) -> Tuple[float, float, float, int]:
    """Wilkinson nice-numbers bin sizing.

    Snaps the bin width to the nearest 1/2/5 × 10^n, then extends min/max
    outward to the nearest step boundary.

    Args:
        min_step: Floor on the resulting step. Pass 1.0 for integer-valued
            data so a single bin never splits two adjacent integers apart.
            When this floor brings the step down to exactly 1, bin edges are
            aligned to whole numbers (one bin per integer value).

    Returns:
        (step, new_min, new_max, new_num_bins)
    """
    raw_step = (data_max - data_min) / num_bins
    if raw_step == 0:
        step = max(1.0, min_step)
        if min_step > 0 and step == 1.0:
            new_min = math.floor(data_min)
            new_max = math.ceil(data_max) + 1
            return step, float(new_min), float(new_max), int(new_max - new_min)
        half = step / 2
        return step, data_min - half, data_min + half, 1
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = min([1, 2, 5], key=lambda m: abs(raw_step - m * magnitude)) * magnitude
    step = max(step, min_step)

    if min_step > 0 and step == 1.0:
        new_min = math.floor(data_min)
        new_max = math.ceil(data_max) + 1
        new_num_bins = int(new_max - new_min)
        return step, float(new_min), float(new_max), new_num_bins

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

    def compute(
        self, x: np.ndarray, num_bins: int, *, min_step: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a federated histogram with Wilkinson-snapped bin boundaries.

        Args:
            x: Local 1D data array for this client.
            num_bins: Hint for the desired number of bins; actual count may differ
                after snapping to a nice step size.
            min_step: Floor on the bin step. Pass 1.0 for integer-valued
                columns so bins use whole-number edges (one bin per integer).

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
            float(min_val), float(max_val), num_bins, min_step=min_step
        )

        x_clean = x[~np.isnan(x)]
        counts, bin_edges = np.histogram(
            x_clean, bins=new_num_bins, range=(new_min, new_max)
        )
        counts = self.agg.fed_sum(counts.astype(float))
        return counts, bin_edges


GroupedHistogramPayload = Dict[
    str, Union[List[HistogramBin], List[List[Optional[int]]]]
]


class HistogramResult:
    def __init__(
        self,
        bins: List[HistogramBin],
        counts: List[Optional[int]],
        grouped: Dict[str, GroupedHistogramPayload],
    ):
        self.bins = bins
        self.counts = counts
        self.grouped = grouped

    def as_payload(self) -> Dict[str, object]:
        return {
            "bins": self.bins,
            "counts": self.counts,
            "grouped": self.grouped,
        }


class FederatedGroupedHistogram:
    """
    Improved federated histogram using discovery-based categorization.

    Always discovers actual categories from data (via CategoricalHistogram),
    ensuring no values are silently dropped when metadata enumerations
    are outdated or incomplete. Supports both numerical (Wilkinson/Simple)
    and categorical histograms with optional grouping.
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client

    def histogram(
        self,
        *,
        data: pd.DataFrame,
        y_var: str,
        x_vars: List[str],
        metadata: Dict,
        bins: int = 20,
        histogram_type: str = "wilkinson",
        is_integer: bool = False,
        min_row_count: int,
    ) -> HistogramResult:
        y_meta = metadata.get(y_var, {})
        is_categorical = y_meta.get("is_categorical", False)

        if is_categorical:
            return self._categorical_histogram(
                data=data,
                y_var=y_var,
                x_vars=x_vars,
                metadata=metadata,
                min_row_count=min_row_count,
            )

        return self._numerical_histogram(
            data=data,
            y_var=y_var,
            x_vars=x_vars,
            metadata=metadata,
            bins=bins,
            histogram_type=histogram_type,
            is_integer=is_integer,
            min_row_count=min_row_count,
        )

    def _categorical_histogram(
        self,
        *,
        data: pd.DataFrame,
        y_var: str,
        x_vars: List[str],
        metadata: Dict,
        min_row_count: int,
    ) -> HistogramResult:
        aggregator = NumpyAggregator(self.agg_client)

        # Discover categories from data via federated union, so that
        # categories missing from (or beyond) the metadata enumeration are
        # never silently dropped.
        cat_hist = CategoricalHistogram(aggregator)
        cats, discovered_counts = cat_hist.compute(data[y_var].to_numpy(dtype=object))
        discovered = cats.tolist()
        counts_by_category = dict(zip(discovered, discovered_counts))

        # Report the full metadata-declared enumeration (in its declared
        # order), with unobserved categories carrying a true zero count, plus
        # any extra categories discovered that the enumeration doesn't list.
        enumerations = metadata.get(y_var, {}).get("enumerations")
        if enumerations:
            enum_order = list(enumerations.keys())
            categories = enum_order + [c for c in discovered if c not in enum_order]
        else:
            categories = discovered

        global_counts = [counts_by_category.get(c, 0.0) for c in categories]
        masked = self._mask_counts(global_counts, min_row_count)

        grouped = {}
        for x_var in x_vars:
            groups = self._get_groups(aggregator, metadata, x_var, data[x_var])
            n_g, n_c = len(groups), len(categories)
            matrix = np.zeros((n_g, n_c), dtype=float)
            y_str = data[y_var].astype(str)
            x_str = data[x_var].astype(str)
            for i, group in enumerate(groups):
                subset_y = y_str[x_str == str(group)]
                matrix[i] = [float((subset_y == str(c)).sum()) for c in categories]
            global_matrix = np.asarray(
                aggregator.fed_sum(matrix.flatten()), dtype=float
            ).reshape(n_g, n_c)
            grouped[x_var] = {
                "groups": [str(g) for g in groups],
                "counts": [
                    self._mask_counts(row, min_row_count) for row in global_matrix
                ],
            }

        return HistogramResult(
            bins=[str(c) for c in categories],
            counts=masked,
            grouped=grouped,
        )

    def _numerical_histogram(
        self,
        *,
        data: pd.DataFrame,
        y_var: str,
        x_vars: List[str],
        metadata: Dict,
        bins: int,
        histogram_type: str,
        is_integer: bool,
        min_row_count: int,
    ) -> HistogramResult:
        aggregator = NumpyAggregator(self.agg_client)

        values = data[y_var].to_numpy(dtype=float, copy=False)
        bins = max(1, int(round(bins)))

        is_integer = is_integer or metadata.get(y_var, {}).get("sql_type") == "int"

        if histogram_type == "wilkinson":
            hist_algo = WilkinsonHistogram(aggregator)
            global_hist, bin_edges = hist_algo.compute(
                values, bins, min_step=1.0 if is_integer else 0.0
            )
        else:
            hist_algo = SimpleHistogram(aggregator)
            global_hist, bin_edges = hist_algo.compute(values, bins)
        masked = self._mask_counts(global_hist, min_row_count)

        grouped = {}
        for x_var in x_vars:
            groups = self._get_groups(aggregator, metadata, x_var, data[x_var])
            n_g, n_b = len(groups), len(bin_edges) - 1
            matrix = np.zeros((n_g, n_b), dtype=float)
            for i, group in enumerate(groups):
                mask = data[x_var].astype(str) == str(group)
                subset = data.loc[mask, y_var].to_numpy(dtype=float, copy=False)
                local_hist, _ = np.histogram(subset, bins=bin_edges)
                matrix[i] = local_hist.astype(float)
            global_matrix = np.asarray(
                aggregator.fed_sum(matrix.flatten()), dtype=float
            ).reshape(n_g, n_b)
            grouped[x_var] = {
                "groups": [str(g) for g in groups],
                "counts": [
                    self._mask_counts(row, min_row_count) for row in global_matrix
                ],
            }

        return HistogramResult(
            bins=bin_edges.tolist(),
            counts=masked,
            grouped=grouped,
        )

    def _get_groups(self, aggregator, metadata, x_var, series):
        """Return groups for a grouping variable.

        Uses metadata enumerations when available so that all workers agree on the
        same ordered list without a network round-trip. Falls back to a federated
        union when enumerations are absent, ensuring consistency across workers.
        """
        x_meta = metadata.get(x_var, {})
        if x_meta and x_meta.get("enumerations"):
            return list(x_meta["enumerations"].keys())
        local_groups = series.dropna().astype(str).unique()
        global_groups = aggregator.fed_union(np.array(local_groups, dtype=object))
        return sorted([str(g) for g in global_groups])

    def _mask_counts(self, values, min_row_count: int):
        """Replace counts below *min_row_count* with ``None`` (privacy mask)."""
        result = []
        for v in values:
            c = int(round(float(v)))
            result.append(c if c >= min_row_count else None)
        return result
