# Quartiles (Histogram-Based)

Runtime wrapper: [statistics/quartiles.py](../../exaflow/algorithms/exareme3/statistics/quartiles.py).

## Inputs

- `y`: one numerical variable, integer or real.
- `num_bins`: integer from 2 to 100, default 20.

Quartile levels are fixed at 0.25, 0.5, and 0.75. Integer mode is derived from
the selected variable's metadata. Refinement depth is internal; this request
does not expose `max_iterations` or `threshold` parameters.

## Method

See [percentiles and quartiles](../../exaflow/algorithms/federated/docs/quartiles.md)
for the library interface, histogram refinement, achieved cumulative fraction,
and approximation limits. The wrapper calls `Percentile.compute` once per
quartile using the same aggregation session.

## Outputs

`quantiles` contains three records, each with `q`, `value`, and `actual_q`.
`actual_q` is null when the achieved cumulative fraction matches the target;
otherwise it reports the achieved fraction. When the library returns no estimate,
both `value` and `actual_q` are null.
