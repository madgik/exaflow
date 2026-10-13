# Histogram (`feta_histogram`)

## Overview and inputs

Computes histogram counts for one numerical or nominal target in `y`, optionally
stratified separately by categorical variables in `x`. Both groups arrive as
`FedColumns`, preserving SQL types, categorical flags and category order. Missing
values are rejected; preprocessing is not supported. Initialization skips workers
below their configured privacy minimum.

`bins` ranges from 1 to 100 and defaults to 20. `histogram_type` is `simple`
(default) or `wilkinson`. These settings apply only to numerical targets.

## Statistical method

Simple numerical histograms use equal-width bins between global extrema. Constant
inputs use `(value, value + 1)` as the range. Wilkinson binning snaps widths to
1/2/5 times a power of ten and extends boundaries outward. Integer measurements
have a minimum Wilkinson step of one. The actual Wilkinson bin count can differ
from the requested count. NumPy counting uses left-closed, right-open bins, except
the final bin includes the upper edge.

Categorical targets preserve metadata enumeration order, including unobserved
categories. Text labels are discovered through global union and any additional
observed labels follow declared categories. Integer codes are counted from
metadata; unknown codes raise an input error, following the approved guarantee
that metadata contains every valid categorical code. Numeric unique remains
restricted. Grouping categories use metadata order or observed text union when
metadata is absent.

## Computation without row-level data

Numerical extrema determine shared edges. The new `fnp.histogram` local operation
computes unweighted bin counts, automatically summed through the aggregation
server. Categorical counts use equality and global sums. Grouped histograms filter
worker-local observations and repeat counting with the same category order or
numerical edges. Raw observation tensors remain local.

### Federated flow

```text
prepare selected columns and metadata
if target is categorical:
    resolve ordered codes
    counts = [sum(target == code) for code in codes]
else:
    lower, upper = min(target), max(target)
    determine simple or Wilkinson boundaries
    counts, edges = histogram(target, bins=count, range=boundaries)
for each grouping variable and ordered group:
    sample = target[grouping == group]
    compute categorical counts or histogram(sample, bins=edges)
mask counts below the strictest participating worker minimum
return overall histogram followed by grouped histograms
clean up execution contexts
```

## Technical decisions and outputs

Workers validate numerical input placement and finite, strictly increasing
literal bin edges before NumPy allocation/counting. Count outputs are int64
additive partials. The initial `fnp.histogram` contract supports integer bin counts
and explicit NumPy edge arrays; weights, density and automatic estimators are
excluded. Concrete numerical arrays execute with NumPy locally.

The response contains `histogram` entries with `var`, `grouping_var`,
`grouping_enum`, `bins` and `counts`. The first entry is ungrouped with null grouping
fields. Numerical bins are edges; categorical bins are stringified codes.
Every count below the privacy minimum becomes null, including zeros. Global
counts are visible to the controller before response masking; this preserves
existing trust boundaries and does not constitute general query privacy.

## Validation and limitations

Simulation tests compare complete responses with the existing implementation for
real/integer measurements, text/integer categories, constants, both binning
strategies, grouping, absent categories and worker exclusion. Kernel tests verify
NumPy parity, boundary inclusion, empty filtered partitions and worker validation.
Package tests exercise Quart, DuckDB, protobuf, workers and aggregation RPCs.
Development deployment compares complete responses with the five existing
histogram validation fixtures using identical non-null filters.

A new HISTOGRAM operation is added to the FETA protobuf enum. Regenerate bindings
and deploy controller/workers together. Existing Exareme3 code and RPCs remain
unchanged. Histogram memory is bounded by configured tensor limits. Integer
categorical metadata must be complete; this deliberately rejects inconsistent
data that the reference implementation can discover.
