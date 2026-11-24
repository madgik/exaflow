# Standard Scaler

## Overview

`standard_scaler` creates one standardized numerical column per selected source,
retaining all sources. It is a preprocessing step, not a standalone analysis. The derived
columns and their metadata are available to the downstream analysis within the
same request; stored datasets are unchanged.

## Inputs

- `source_variables`: a nonempty ordered list of distinct numerical variables
  available in the current input.
- `new_column_names`: one nonblank, unique new name per source, in the same order.

For example:

```json
{
  "source_variables": ["crp", "subjectage"],
  "new_column_names": ["crp_scaled", "age_scaled"]
}
```

The first output scales `crp`, and the second scales `subjectage`. Output count
comes from the source-list length through `column_count_param="source_variables"`;
there is no separate count parameter.

There is no method selector or recipe parameter. Numerical integer and real
sources are accepted; the generated columns have numerical `REAL` metadata.

## Method

For each observed value, the output is `(x - mean) / scale`. The mean and
population variance (`ddof=0`) are computed independently per source over observations pooled across
participants. Missing values are omitted featurewise from fitting and remain missing in
the output. Infinite values and a source with no observed values are rejected.
Empty local partitions contribute zero counts and sums; they do not prevent
fitting when other participants supply observations.

The variance uses the existing corrected two-pass calculation: aggregate sums
and valid counts, then aggregate squared deviations and residual deviations
around the pooled mean. Constant or numerically indistinguishable scales use
one. This retains variation at tiny magnitudes and avoids the cancellation of
a raw `sum(x*x) - sum(x)**2/n` calculation.

## Computation without row-level data

Only sums, valid counts, squared-deviation sums, and residual-deviation sums
are aggregated. No observation vectors or category values are exchanged.

### Federated flow

```text
Validate source and declared output metadata.
Aggregate local sums and valid counts; compute the pooled mean.
Aggregate deviations and squared deviations; compute the population scale.
Retain the validated learned parameters on the fitted transformer.
Transform the local sources and append derived columns in the declared order.
Return transformed data and metadata through the existing preprocessing path.
Run the selected downstream analysis on its requested columns.
```

## Technical decisions

The mathematical implementation is
`exaflow.algorithms.federated.feature_engineering.FederatedStandardScaler`.
`fit()` returns `None` and retains the fitted state on that instance; subsequent `transform()` calls reuse its
`mean_`, `scale_`, and `feature_names_in_` without additional aggregation.
Input names/order, finite means, and positive finite scales remain validated.
The mathematical core does not mutate the input DataFrame. The runtime step
appends its outputs to the invocation-owned DataFrame without replacing the
sources. Existing pipelines and column-composition code are unchanged.

Runtime hooks assume the normal validated request path: parameter types and
membership are checked by the controller, output names by the shared output
contract, and the aggregation client is supplied from the declared component.
The wrapper retains numerical-metadata and distinct-source checks, which are
not expressed by the parameter enum. Direct calls bypassing the runtime path
do not receive additional wrapper collision/client guards. The mathematical
class still validates its data, schema, and fitted attributes independently.

### Privacy policy

The step relies on the existing worker and downstream-analysis privacy policies.
Where enabled by the UDF, the shared worker check enforces
`privacy.minimum_row_count` on participant row counts before and after
preprocessing. It is not a per-feature non-missing or per-dataset floor. Reports
such as `describe` retain their own exemptions and summary masking.

The scaler does not add a separate count threshold or silently exclude small
groups from the pooled fit. A source with no observed values anywhere remains
mathematically invalid. Missing local values and empty partitions contribute
neutral statistics whenever the execution path permits them.

This intentionally removes the pilot's additional non-missing/group floor.
Downstream masking does not imply that masked groups had no influence on fitted
scaling parameters; paired source/derived summaries can reveal those parameters.
This contract is not a differential-privacy guarantee, and compositions requiring
stronger influence protection need a separately reviewed privacy policy.

## Outputs

The step returns transformed data and metadata internally, not a separate
analysis response. `new_column_names[i]` identifies the numerical column derived
from `source_variables[i]`.
Its API visibility follows the selected downstream algorithm's normal output.
No fitted recipe or raw transformed observations are added to the HTTP response.

## Validation

Focused tests compare pooled statistics and transformed values with
[scikit-learn StandardScaler](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html),
including uneven partitions, large offsets, tiny values, constant features,
missing values, and empty participants. Lifecycle tests cover `None` return with fitted attributes, unfitted errors,
reuse on new data, invalid fitted attributes, and native composition. Runtime
tests cover registration, output contracts, source preservation, and worker
preprocessing execution. Policy regressions verify the common worker row floor
is still enforced and the scaler adds no per-dataset/non-missing floor or silent
group exclusion. HTTP tests cover one/multiple sources, single-dataset requests,
existing report masking, count mismatches, and recovery after an empty fit.

## Limitations and contract change

Each source has exactly one output. This pilot does not add a multi-stage
federated pipeline implementation or persistent fitted-model storage.
Fitted-object reuse is within the caller's Python execution; state is not
automatically retained for later HTTP requests.

The scalar `source_variable` parameter is replaced by `source_variables`, even
for one source. Chaining the concrete transformer's `fit()` return is no longer
supported. Use a retained instance, call `fit()`, then `transform()`; no aliases
or legacy request normalization are provided.

The previous mathematical import `federated.preprocessing.StandardScaler` and
its recipe replay API are removed rather than aliased. Use the canonical
`federated.feature_engineering.FederatedStandardScaler` with the native named-DataFrame
arguments. Other transformation primitives and PCA/KMeans reusable recipes
are unchanged. This pilot exposes `standard_scaler`, not the unmerged combined
`variable_scaler` request contract.
