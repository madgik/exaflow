# Standardized mean difference (`feta_standardized_mean_difference`)

## Overview and inputs

Computes pairwise Cohen's d for a numerical outcome `y` across nominal groups
in `x`. The grouping input is a typed `FedColumns` collection; numerical outcomes
are `FedTensor` values. Text and integer categorical codes are supported.
Missing values are rejected and preprocessing is not supported. Workers below
their configured privacy minimum are skipped during initialization.

## Statistical method

For group g, compute count n, mean m and sample variance
`v = sum((y - m)**2) / (n - 1)`. For each eligible pair:

```text
pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
smd = (m1 - m2) / sqrt(pooled)
```

A nonpositive pooled variance returns 0. Groups must have at least two
observations and meet the highest configured privacy minimum among participating
workers. These are global group counts, not per-worker group counts. With
uniform worker thresholds this preserves the existing algorithm's behavior;
heterogeneous thresholds use the stricter policy approved for FETA.

## Computation without row-level data

Text group labels use global string union. Integer group codes come from declared
metadata and global equality counts; unobserved codes are omitted. Undeclared
integer codes raise an explicit input error, preserving the current FETA
categorical-input contract without releasing arbitrary numeric observations.
Groups are ordered by their string representation, including integer labels,
to preserve the reference algorithm's pair order and effect-size sign.

### Federated flow

```text
initialize inputs and receive worker privacy minima
minimum = max(eligible worker minima, 2)
discover observed text labels or count declared integer codes
for each group ordered by str(code):
    sample = outcome[grouping == code]
    n = len(sample)
    skip if n < minimum
    mean = sum(sample) / n
    variance = sum((sample - mean)**2) / (n - 1)
for each eligible unordered group pair:
    compute pooled variance and Cohen's d on the controller
clean up all execution contexts
```

## Technical decisions and outputs

Uses existing equality, filtering, arithmetic, length and sum operations. Centered
squared deviations avoid the cancellation associated with subtracting squared
means from second moments. The result contains `comparisons`, with `group1`,
`group2` and `smd` for each eligible pair. No eligible pairs produces an empty
list; integer category codes remain integers.

Initialization adds a required-to-be-present `minimum_group_size` response
field. Deploy the regenerated protobuf bindings, controller and workers together;
older workers lacking the field are rejected rather than assigned a weaker
privacy default. Existing Exareme3 RPCs and implementations are unaffected.

## Validation and limitations

Simulated federation tests compare with the existing sufficient-statistics
implementation across partitions, text/integer labels, privacy thresholds,
constant samples, empty pair sets and heterogeneous worker minima. Package
integration exercises the real analysis route, DuckDB and gRPC services. Live
development-deployment validation compares complete responses against the
existing algorithm with identical non-null filters.

Group counts and sufficient statistics become global values. Minimum group sizes
constrain comparison release but do not establish a general privacy guarantee
against repeated or overlapping queries. Integer categorical inputs require
complete declared codes; text categories are discovered from observed data.
