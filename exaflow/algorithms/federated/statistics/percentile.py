from typing import Optional
from typing import Tuple

import numpy as np
from pandas.core.series import Series

from exaflow.algorithms.federated.statistics.histogram import SimpleHistogram
from exaflow.algorithms.federated.statistics.histogram import WilkinsonHistogram
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


class Percentile:
    def __init__(self, aggregator: NumpyAggregator):
        self.agg = aggregator
        self.simple_hist = SimpleHistogram(aggregator)
        self.wilkinson_hist = WilkinsonHistogram(aggregator)

    def compute(
        self,
        x: Series,
        q: float,
        *,
        num_bins: int = 20,
        max_iterations: int = 5,
        threshold: float = 0.001,
        is_integer: bool = False,
    ) -> Optional[Tuple[float, Optional[float]]]:
        if is_integer:
            result = self._compute_integer(x, q, num_bins, max_iterations)
        else:
            result = self._compute(x, q, num_bins, max_iterations, threshold)
        if result is None:
            return None
        value, actual_q = result
        if abs(actual_q - q) <= 1e-9:
            return value, None
        return value, actual_q

    def _fed_count_at_or_below(
        self, x: np.ndarray, *, lower: float, value: float
    ) -> float:
        """Federated count of points in [lower, value] across all workers.

        Each worker counts its local elements that fall in the closed interval
        ``[lower, value]`` and the scalar counts are summed across the federation.
        ``lower`` scopes the count to the current sub-range; NaNs are excluded
        naturally since comparisons against them are False.
        """
        local_count = float(np.count_nonzero((x >= lower) & (x <= value)))
        return float(self.agg.fed_sum(np.array([local_count]))[0])

    def _compute(
        self,
        x: Series,
        q: float,
        num_bins: int,
        max_iterations: int,
        threshold: float,
    ) -> Optional[Tuple[float, float]]:
        # Constant (zero-range) data: the histogram would widen [v, v] to
        # [v, v+1] and the recursion would interpolate slightly above v, so
        # return the exact value up front. Because the examined subset is pushed
        # into each round, this also catches a sub-range that collapses to a
        # single repeated value during refinement. The min/max are reused as the
        # histogram range, so no extra federated calls are introduced.
        min_val = float(self.agg.global_min(x.values))
        max_val = float(self.agg.global_max(x.values))
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return None
        if min_val == max_val:
            return min_val, (0.0 if q == 0 else 1.0)
        try:
            bin_count, bin_edges = self.simple_hist.compute(
                x.values, num_bins, bounds=(min_val, max_val)
            )
        except BadInputError:
            return None

        n = int(sum(bin_count))
        if n == 0 or q == 0:
            return None if n == 0 else (float(bin_edges[0]), 0.0)
        total_counter = 0
        for i, count in enumerate(bin_count):
            total_counter += count
            if abs(total_counter / n - q) < threshold:
                return bin_edges[i + 1], total_counter / n
            if total_counter / n > q:
                total_counter -= count
                new_q = (q * n - total_counter) / count
                if max_iterations > 0:
                    # Push only the selected bin's data into the next round so it
                    # is re-binned over its own extent (no zoom_bounds needed).
                    # Mirror np.histogram membership: half-open [lo, hi) for every
                    # bin except the last, which is closed [lo, hi].
                    lo, hi = bin_edges[i], bin_edges[i + 1]
                    if i == len(bin_count) - 1:
                        sub = x[(x >= lo) & (x <= hi)]
                    else:
                        sub = x[(x >= lo) & (x < hi)]
                    result = self._compute(
                        sub, new_q, num_bins, max_iterations - 1, threshold
                    )
                    if result is None:
                        return None
                    value, sub_q = result
                    return value, total_counter / n + sub_q * (count / n)
                else:
                    value = bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) * new_q
                    # Real empirical CDF at `value`: directly count the data
                    # points within this sub-range that fall at/below it, instead
                    # of assuming a uniform spread inside the bin.
                    below = self._fed_count_at_or_below(
                        x.values, lower=bin_edges[0], value=value
                    )
                    return value, below / n
        # The cumulative never exceeded q (e.g. q == 1.0): clamp to the max.
        return float(bin_edges[-1]), total_counter / n

    def _compute_integer(
        self,
        x: Series,
        q: float,
        num_bins: int,
        max_iterations: int,
    ) -> Optional[Tuple[float, float]]:
        """Locate an integer-valued percentile via whole-number Wilkinson bins.

        The bucket holding the target rank is re-binned at finer resolution until
        it spans a single integer -- the discrete q-quantile, i.e. the smallest
        value with ``CDF(value) >= q``. A very wide range may not narrow to a
        single integer within ``max_iterations``, in which case the result is an
        estimate.

        ``actual_q`` is always the value's true empirical CDF -- the global
        fraction of points at or below it. The public ``compute`` then reports
        ``None`` when this equals the requested ``q`` (an exact landing).
        """
        data = x.values
        try:
            bin_count, bin_edges = self.wilkinson_hist.compute(
                data, num_bins, min_step=1.0
            )
        except BadInputError:
            return None

        n = int(sum(bin_count))
        if n == 0 or q == 0:
            return None if n == 0 else (float(bin_edges[0]), 0.0)

        global_lower = float(bin_edges[0])
        value: Optional[float] = None
        passed = 0.0  # global count strictly below the current sub-range
        for _ in range(max_iterations + 1):
            cum = passed
            for i, count in enumerate(bin_count):
                cum += count
                if cum / n >= q:  # bucket i reaches the target rank
                    lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
                    if hi - lo <= 1.0:
                        # Single-integer bucket: the discrete q-quantile.
                        value = lo
                    else:
                        # Re-bin only this bucket's integers, finer next pass.
                        # Mirror np.histogram membership: the last bucket is
                        # closed [lo, hi], every other is half-open [lo, hi);
                        # otherwise a max sitting on the final edge is dropped.
                        passed = cum - count
                        if i == len(bin_count) - 1:
                            sub = data[(data >= lo) & (data <= hi)]
                        else:
                            sub = data[(data >= lo) & (data < hi)]
                        try:
                            bin_count, bin_edges = self.wilkinson_hist.compute(
                                sub, num_bins, min_step=1.0
                            )
                        except BadInputError:
                            value = lo  # approximate: cannot refine further
                    break
            else:
                value = float(bin_edges[-1])  # q past the data: clamp to max
            if value is not None:
                break
        if value is None:
            # Iterations exhausted on a still-wide bucket: take its low integer.
            value = float(bin_edges[0])

        actual_q = (
            self._fed_count_at_or_below(data, lower=global_lower, value=value) / n
        )
        return value, actual_q
