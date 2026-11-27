# Histograms

Implementation: [statistics/histogram.py](../statistics/histogram.py).

Numerical and categorical histogram primitives aggregate aligned local counts. A grouped facade adds column metadata, grouping, and caller-configured suppression.

## Library interface

`SimpleHistogram(aggregator).compute(x, num_bins, *, bounds=None)` and `WilkinsonHistogram(aggregator).compute(x, num_bins, *, min_step=0.0)` return `(counts, edges)`. `CategoricalHistogram(aggregator).compute(x)` returns aligned counts and labels. These primitives use `NumpyAggregator`.

`FederatedGroupedHistogram(agg_client).histogram(data=..., y_var=..., x_vars=..., metadata=..., bins=20, histogram_type="wilkinson", is_integer=False, min_row_count=...)` returns `HistogramResult(bins, counts, grouped)`. The grouped facade takes caller-supplied column metadata and a suppression threshold. Its default strategy is Wilkinson.

## Statistical method

For numerical variables, the algorithm supports two binning strategies:

- `simple`: equal-width bins between the aggregated minimum and maximum.
- `wilkinson`: bin boundaries are snapped to readable 1/2/5 x 10^n step sizes,
  and the range is extended outward to those boundaries.

For integer-valued numerical variables with `wilkinson` binning, the step size
is floored at `1.0`, so adjacent integer values are not split by sub-integer bin
edges.

For categorical variables, bins are category labels. Metadata enumerations are
reported first in their declared order. Any observed categories not present in
metadata are appended. When no enumeration is available, globally observed
categories are discovered and sorted.

When grouping variables are provided, each grouping level receives a histogram
using the same numerical bin edges or categorical bin order as the base
histogram.

## Federated computation

The histogram is computed without sharing row-level data. Each site contributes
local extrema, category levels, and bin-count vectors. Aggregated quantities are
used to build aligned bins and total counts.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Numerical minimum and maximum | Define shared numerical bin ranges. |
| Category levels | Align categorical bins across sites. |
| Group levels | Align grouped histogram rows. |
| Bin counts | Compute aggregate histogram counts. |
| Group-specific count matrices | Compute counts for each group level and bin. |

### Federated flow

```text
Receive data, variable names, column metadata, and a count threshold.
For categorical y, align declared and observed category levels.
For numerical y, aggregate extrema and define shared simple or Wilkinson edges.
Count local observations in the shared bins.
For each grouping variable, align its levels and count group-by-bin matrices.
Sum base and grouped counts across participants.
Apply the supplied suppression threshold in the grouped facade.
Return bins, base counts, and grouped counts.
```

## Numerical decisions

- Numerical `bins` is rounded to an integer and bounded by the public
  specification range.
- If a simple numerical histogram has identical minimum and maximum, the maximum
  is increased by `1.0` to create a valid bin range.
- Wilkinson binning may return a different number of bins than requested because
  the requested value is treated as a hint for selecting readable boundaries.
- Category and group metadata enumerations preserve their declared order.
- Counts lower than the minimum row-count threshold are returned as `null`.
- The primitives return exact aggregate counts for their chosen bins. Suppression belongs to `FederatedGroupedHistogram` and uses its explicit `min_row_count` argument.

## Reference comparison

Numerical counting is aligned with `numpy.histogram` after shared bin edges are
defined. Standalone tests cover simple numerical histograms, Wilkinson binning,
integer-aware Wilkinson boundaries, categorical histograms, and grouped
histograms.

## Limitations

- The output reports counts, not densities or percentages.
- Numerical bin edges are derived from aggregated extrema, so extreme values can
  affect bin width.
- Counts below the privacy threshold are masked and cannot be distinguished from
  absent low-count bins in the grouped result.
- Grouped histograms can be sparse when many grouping levels are selected.
- Wilkinson binning prioritizes readable boundaries over preserving the exact
  requested number of bins.
