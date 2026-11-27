# Percentiles and quartiles

Implementation: [statistics/percentile.py](../statistics/percentile.py).

`Percentile(aggregator).compute(x, q, *, num_bins=20, is_integer=False)` estimates
a percentile of a pandas Series distributed across participants. The aggregator
is a `NumpyAggregator`; `q` is in `[0, 1]` and `num_bins` is a positive integer.
Quartiles are obtained by calling it with `q=0.25`, `0.5`, and `0.75`.

## Result

The result is `None` when the computation has no usable histogram/range, or
`(value, actual_q)`. `actual_q` is the empirical cumulative fraction at or below
the returned value; it is replaced by `None` when it matches `q` within `1e-9`.
Repeated values can make that fraction exceed the requested percentile.

## Statistical method

The method locates the bin containing the target rank, restricts each local
partition to that bin, and repeats with the rank adjusted to the retained
population. Real-valued input uses equal-width bins and can interpolate when
the refinement budget is exhausted. Integer mode uses whole-number Wilkinson
edges and seeks the smallest integer with cumulative fraction at least `q`.
Both delegate counting/bin construction to the shared [histograms](histogram.md).

### Federated flow

```text
Validate q, num_bins, and is_integer.
Build shared bin edges from aggregated extrema and sum local bin counts.
Locate the bin containing the requested cumulative rank.
Return a matching boundary when possible.
Otherwise keep only that bin's local observations and refine its range/rank.
For a final estimate, count observations at or below the chosen value.
Reconstruct the cumulative fraction in the original population.
Return the estimate and achieved fraction, or None when no range is usable.
```

## Numerical decisions and limitations

- The refinement budget is internal (`_MAX_ITERATIONS=5`), not a public option.
- Constant real-valued ranges return the repeated value without interpolation.
- Real mode returns `None` for nonfinite global extrema; it does not promise
  to discard infinities and estimate from the finite subset.
- Integer mode converts only the histogram's `HISTOGRAM_FINITE_VALUES_REQUIRED`
  diagnostic into `None`. Other failures propagate.
- Missing observations do not satisfy interval comparisons during refinement.
  Callers should prepare inputs according to the intended population.
- Estimates can differ from NumPy/pandas interpolation conventions, especially
  for duplicates and large gaps. The achieved fraction describes the returned
  estimate rather than claiming exact reference-package quantiles.
- Refinement requires repeated aggregation; `num_bins` trades communication
  volume against resolution. Local rows remain local.
