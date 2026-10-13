# Federated local computation library review

Reviewed **2026-10-13**, initially for all 17 statistics computation modules
and extended to the full
[`algorithms/federated`](../../exaflow/algorithms/federated) tree. The statistics
catalogue is preserved below; the remaining-family review, additional local
computations and difficult language requirements follow it.
This is a source-based catalogue and implementation proposal, not a newly
implemented operation allowlist. Existing FETA decisions remain governed by
[AGENTS.md](AGENTS.md).

The proposed library has three layers: small array primitives, compositions of
those primitives, and controlled dataset-summary operations. Mathematical
postprocessing of completed global results belongs in the controller.

## Migration Status

Runtime counterparts implemented and validated so far are
`feta_pearson_correlation`, `feta_ttest_onesample`, `feta_ttest_paired` and `feta_ttest_independent`.
Independent t-test adds string-label inputs, equality, column selection and
boolean row indexing. Its inference uses existing numerical reductions.
The catalogue below describes requirements, not completion of the remaining
algorithms. Migration proceeds one algorithm at a time with a user continuation
decision after each validated implementation.

## Existing FETA foundation

The current worker allowlist is `ADD`, `SUBTRACT`, `MULTIPLY`, `DIVIDE`, `POWER`,
`TRANSPOSE`, `SUM`, `MATMUL`, `LENGTH`, `EQUAL`, `COLUMN` and `FILTER`. Operands are context-scoped references,
numeric/string/boolean Python values, or supported numerical/boolean NumPy arrays.
Tensors have rank at most two and support numerical, boolean and Unicode dtypes;
object arrays remain rejected. Initial numerical inputs are float64, text inputs
retain labels, and missing values remain rejected.

Arithmetic and transpose preserve a distributed tensor. Reducing its partitioned
axis or contracting aligned distributed matrix dimensions automatically sums
local contributions. Only global **sum** exists in the FETA protocol today.
The shared legacy aggregation service's other operations do not automatically
make them available through FETA.

The public `exaflow.algorithms.feta.federated_numpy` package exposes the existing
sum, transpose and equality operations, including `fnp.sum(a, axis=None)` and `fnp.transpose(a)`.
Import it with `from exaflow.algorithms.feta import federated_numpy as fnp`.
Arithmetic and matrix products continue to use operators. Tensor methods remain
available, so the equivalent method expressions in this catalogue are valid.
The function package also handles concrete real numerical NumPy arrays locally.
Other functions listed below remain proposals until their operations are added.

Counts of rows, sums, squares, centered squared deviations, pairwise differences,
and matrix products are already expressible for prepared, aligned numerical
inputs. A vector dot product can be expressed as `(x * y).sum()`; it does not
require adding one-dimensional `@` support or a `cross_product()` operation.

## Proposed primitive catalogue

Names below describe fixed engineer-controlled operations, not arbitrary NumPy
function names accepted over the wire. Signatures describe worker-local kernels;
`Array` means a stored local array, not a developer-facing wrapper. Parameters
must be typed and bounded. Operations still use one `LocalOperation` class and
one operation per request. Groups of related names are listed together for
readability, not as operation subclasses or batches.

| Family | Local computation contract | Consumers | FETA status |
| --- | --- | --- | --- |
| Arithmetic | `add/subtract/multiply/divide/power(x, other)`; validated broadcasting | Pearson, t-tests, ANOVA, moments, SMD | Implemented |
| Transpose and matrix product | `transpose(x)`, `matmul(x, y)`; 2-D products, aligned partitions | Pearson, two-way ANOVA through OLS | Implemented |
| Sum and row length | `sum(x, axis)`, `length(x)` | All numeric moment calculations | Implemented for ordinary sums and axis-zero distributed length |
| Absolute value | `absolute(x) -> Array` | Mean absolute deviation, MAD outliers, OLS absolute residuals | Add |
| Column selection | `take_columns(x, indices) -> Array`; ordered, bounded numeric indices | Multivariable summaries, design matrices | Implemented for a single column using [:, index]; multiple-column indexing remains open |
| Shape utilities | `reshape/squeeze(x, shape) -> Array`; initially vector ↔ single-column matrix only | Scalar-sample algorithms, OLS response vectors | Add only restricted transforms preserving record alignment |
| Concatenation and constants | `concatenate(arrays, axis)`, `ones_like_rows(x)` | ANOVA intercept/design assembly; pooled Mann–Whitney bounds | Add; column assembly needed, row concatenation can be avoided for pooled bounds |
| Comparisons | `equal/not_equal/less/less_equal/greater/greater_equal(x, value) -> bool Array` | Group selection, percentile intervals, outlier tails | Equality implemented with boolean results; other comparisons remain proposals |
| Boolean logic | `logical_and/or/not(mask) -> bool Array` | Interval membership, missing-value and finite masks | Add; masks must carry row lineage |
| Missing and finite masks | `is_missing(x)`, `is_finite(x)` | Describe, categorical tables, histogram, binned Mann–Whitney | Add; missing differs from non-finite |
| Conditional replacement | `where(mask, x, replacement) -> Array` | Missing-aware sums/extrema and derived moments | Add with shape/dtype validation |
| Row selection | `filter_rows(x, mask) -> Array` | Independent t-test, grouped summaries, percentile refinement | Implemented using x[mask]; same mask object preserves paired alignment |
| Count reductions | `count_nonzero(mask, axis)`, `count_valid(x, axis)` | Category counts, outliers, percentile CDF, missing-aware moments | Add, or compose boolean cast plus sum; row length is not valid-value count |
| Minimum and maximum | `minimum/maximum(x, axis, missing_policy)` | Range, one-way ANOVA, describe, numeric histogram, percentile, binned Mann–Whitney | Add local kernels and global min/max |
| Missing-aware sum | `sum_valid(x, axis)` plus valid counts | Primitive statistics, SMD, describe | Add or compose masks, replacement and ordinary sum; retain all-missing information |
| Local unique values | `unique_nonmissing(x) -> category vector` | Cross tables, categorical/grouped histograms, grouped SMD | Add plus deterministic global union/discovery |
| Category encoding | `encode_categories(x, levels, unknown_policy) -> integer Array` | Categorical grouping and treatment coding | Add; typed strings or worker-side encoding require initialization support |
| Category counts | `category_counts(x, levels, missing_policy) -> count vector` | Describe, categorical histograms | Add; supplied levels define a common output order |
| Contingency counts | `contingency_counts(a, b, levels_a, levels_b, missing_policy) -> count matrix` | Cross-tabulation, chi-squared, Fisher, grouped categorical histogram | Add; fixed ordered level dictionaries |
| Numeric bin counts | `histogram_counts(x, edges, missing_policy) -> count vector` | Histograms, percentile, binned Mann–Whitney | Add; edges chosen globally and identical on every worker |
| Grouped numeric bin counts | `grouped_histogram_counts(x, groups, levels, edges) -> count matrix` | Grouped numeric histograms | Compose group masks + histogram; a dedicated kernel is optional |
| Exact local quantile | `quantile_local(x, q, interpolation, missing_policy)` | Dataset describe quartiles; IQR, MAD and quantile outlier bounds | Add internal kernel with an approved summary-release path; never sum local quantiles |
| Local mean and sample standard deviation | `mean_local(x)`, `std_local(x, ddof)` | Dataset describe; Gaussian outlier bounds | Compose local counts/sums/deviations; require explicitly local summary semantics |
| Numeric cleaning | `numeric_valid_values(x, coercion_policy) -> Array` | Outlier report's numeric conversion and missing removal | Prefer typed initialization for numeric data; preserving coercion requires explicit policy |

`is_missing` must cover the relevant numerical NaN and categorical None/NaN
representations. `is_finite` additionally excludes positive and negative infinity.
The existing binned Mann–Whitney implementation drops all non-finite values;
histogram kernels remove NaNs, and outlier cleaning removes missing/coercion
failures. A shared helper must not silently make these populations identical.

## Reusable compositions

These computations need no additional generic function dispatcher. They can be
composed from the primitive catalogue while reusing stored intermediate tensors.
They are library recipes, not permission to batch successive operations.

| Computation | Local work | Combination and controller work |
| --- | --- | --- |
| Numeric moments | Valid count, sum, sum of squares | Sum each contribution; compute means, raw-moment variance and standard deviation |
| Centered variance | Subtract the global mean, square, sum valid deviations | First aggregate count/sum to obtain the mean; broadcast it; aggregate squared deviations and divide by `n - ddof` |
| Covariance | Preserve caller masking/count semantics, subtract global means, multiply, sum | Sum contributions and divide by the agreed count; preserve the existing caller's missing policy |
| Mean absolute deviation | Subtract global mean, take absolute value, sum | Sum contributions and divide by valid count; distinguish this from median absolute deviation |
| RMS | Square and sum valid values | Sum contributions, divide by valid count, square root on controller |
| Group moments | Encode/group/filter, then count, sum, squared sum or centered deviations | Sum vectors in identical level order; derive group means and variances |
| Treatment coding | Encode factors against declared levels; omit the first reference level; generate float indicators | Intercept, selected columns and pairwise products form five ANOVA designs |
| Interaction columns | Multiply every selected factor-A dummy by every factor-B dummy | Preserve the source design's column order and common row mask |
| OLS sufficient statistics | `X.T @ X`, `X.T @ y`, length, `y.sum()`, `(y * y).sum()` | Sum; pseudoinverse, coefficient solve and matrix rank on controller |
| OLS residual statistics | Broadcast fitted coefficients; compute `r = y - X @ beta`, squared and absolute residual sums | Sum residual statistics; derive model and ANOVA summaries on controller |
| Interval/CDF count | Compare with lower/upper bounds, combine masks, count | Sum counts; percentile refinement decisions happen on controller |
| Outlier bound recipes | Local mean/std, quantiles, or `median(abs(x - median(x)))` | Bounds are per dataset in the existing report; apply tails and privacy rules locally |

Centered variance is a two-stage computation, not a sum of worker variances or
an average of worker means. Its first stage must not discard worker-local data.
For parity, retain the algorithm's existing choice between raw moments and
centered deviations rather than replacing every formula with one recipe.

## Complete source-to-library mapping

The table covers all computation modules, including modules not re-exported
by `statistics/__init__.py`. Aggregation operators describe the existing core,
not current FETA availability.

| Source module | Worker-local requirements | Cross-worker combination | Work after global results |
| --- | --- | --- | --- |
| [pearson_correlation.py](../../exaflow/algorithms/federated/statistics/pearson_correlation.py) | Ordered aligned columns; row length; sums and squared sums; `y.T @ x` | Sum | Coefficient clipping, constant-feature handling, beta p-values, Fisher-z confidence intervals |
| [ttest_onesample.py](../../exaflow/algorithms/federated/statistics/ttest_onesample.py) | Sample flattening; count, sum, squared sum; deviations from supplied mean (also expressible from moments) | Sum | Student t statistic, tails, intervals and Cohen's d |
| [ttest_paired.py](../../exaflow/algorithms/federated/statistics/ttest_paired.py) | Equal aligned samples; differences; counts, sums and squared sums of samples/differences | Sum | Paired t statistic, standard errors, intervals and effect size |
| [ttest_independent.py](../../exaflow/algorithms/federated/statistics/ttest_independent.py) | Group equality masks; selection; per-group count, sum and squared sum | Sum | Pooled Student t statistic, intervals and effect size; current implementation is not Welch |
| [standardized_mean_difference.py](../../exaflow/algorithms/federated/statistics/standardized_mean_difference.py) | Missing-aware count/sum; centered squared deviations; observed group discovery and selection for pairwise mode | Sum; category union for pairwise grouping | Pooled variance and pairwise effect sizes; minimum eligible group size |
| [anova_oneway.py](../../exaflow/algorithms/federated/statistics/anova_oneway.py) | Provided ordered groups; count, sum, squared sum, min and max per group; empty-group identities | Sum, min, max | ANOVA table, Tukey comparisons, summaries and zero-group pruning |
| [anova_twoway.py](../../exaflow/algorithms/federated/statistics/anova_twoway.py) | Complete-case selection; categorical treatment coding, intercept, interactions, five designs; OLS products, prediction and residual sums | Sum | Pseudoinverse/rank/coefficients and Type-I/II sums of squares, F tests |
| [primitive_statistics.py](../../exaflow/algorithms/federated/statistics/primitive_statistics.py) | Valid counts/sums; centered deviations and products; powers; absolute values; min/max | Sum, min, max | Count, mean, variance/std, range, CV, mean absolute deviation, RMS, mean square, covariance and correlation |
| [cross_tabular_table.py](../../exaflow/algorithms/federated/statistics/cross_tabular_table.py) | Category discovery or supplied levels; missing-presence flag; aligned contingency counts including optional missing category | Category union, max (missing presence), sum | Labelled table and all-zero row/column pruning; empty-table error |
| [chi_squared.py](../../exaflow/algorithms/federated/statistics/chi_squared.py) | Same contingency kernel as cross-tabulation | Same cross-table combination | `chi2_contingency`, expected frequencies, degrees of freedom and labels |
| [fisher_exact.py](../../exaflow/algorithms/federated/statistics/fisher_exact.py) | Same contingency kernel as cross-tabulation | Same cross-table combination | Enforce 2×2 shape; Fisher odds ratio/p-value and labels |
| [histogram.py](../../exaflow/algorithms/federated/statistics/histogram.py) | Numeric bin counts, extrema; categorical unique/counts; grouping masks; contingency/grouped-bin matrices | Sum, min, max, category union when discovery is needed | Simple/Wilkinson common edges, metadata category order plus discovered extras, result masking |
| [percentile.py](../../exaflow/algorithms/federated/statistics/percentile.py) | Extrema, common-edge histograms, interval filtering, count at/below candidate; repeated local subsets | Min, max, sum in multiple sequential rounds | Select/refine target bin, interpolate or use integer Wilkinson refinement; actual empirical CDF |
| [binned_mann_whitney_u_test.py](../../exaflow/algorithms/federated/statistics/binned_mann_whitney_u_test.py) | Finite filtering; sample counts; shared pooled bounds; two common-edge histograms | Sum, min, max | Histogram cumulative/mid-ranks, tie correction, U statistic, continuity correction and normal tails |
| [describe.py](../../exaflow/algorithms/federated/statistics/describe.py) | Dataset grouping; per-variable missing policy; valid/missing/total counts; sums, squares, extrema; exact local quartiles/std; category counts; approved local records | Sum, min, max; iterative percentile reductions for global quartiles | Global summaries; histogram-estimated global quartiles/median; retain only eligible dataset-variable populations |
| [descriptive_stats.py](../../exaflow/algorithms/federated/statistics/descriptive_stats.py) | Delegates to describe, grouped histogram and Pearson; no independent kernel | Delegated | Delegated |
| [outlier_report.py](../../exaflow/algorithms/federated/statistics/outlier_report.py) | Dataset grouping; numeric cleaning; exact per-dataset mean/std/quantiles/median/MAD; comparisons and tail counts; threshold validation and suppressed records | None in this core: per-dataset reports are local | Concatenate approved records; do not replace dataset-specific bounds with federation-wide bounds |

Two-way ANOVA's transitive dependency is
[`linear_model/ols.py`](../../exaflow/algorithms/federated/linear_model/ols.py).
Its current OLS path also computes absolute residual sums even though the ANOVA
caller only consumes RSS, observation count and rank. `absolute` is therefore
needed to reuse that path faithfully, but not mathematically necessary for an
ANOVA-only RSS implementation. Intercept construction must account for the
existing OLS `fit_intercept` behavior, not blindly add another intercept.

SMD's grouped path delegates to
[`sql/sql.py`](../../exaflow/algorithms/federated/sql/sql.py), which discovers global
group keys, sorts them by string representation, and uses missing placeholders
for locally absent groups. FETA needs the grouping semantics; it must not expose
that helper's arbitrary expression strings or callable aggregation interface in
worker requests.

## Computations that do not need a worker operation

After their inputs have been globally aggregated, square roots, clipping,
logarithms, inverse hyperbolic tangent, Student/F/normal distribution functions,
Fisher exact and chi-squared tests, Tukey p-values, matrix pseudoinverses/rank,
cumulative histogram counts and mid-ranks can use ordinary NumPy/SciPy on the
controller. Their occurrence in the current worker-authored federated core does
not make them local-data kernels needed by FETA. Sorting or selecting observations
for exact local quantiles is an internal quantile implementation detail; it does
not justify exposing sorted raw arrays or a generic row-download API.

## Required global aggregation capabilities

| Operator | Why needed | Current FETA |
| --- | --- | --- |
| Sum | Counts, sufficient statistics, histograms, contingency matrices and residual sums | Implemented |
| Min | Global extrema and per-group minima | Not implemented |
| Max | Global extrema and per-group maxima; categorical missing-presence flags | Not implemented |
| Deterministic union | Discovered categories/group keys and, in other families, numeric event times/cluster identifiers | Not implemented; requires typed category transport and release rules |

No reviewed algorithm requires a global product, a standalone global average, or a
merge of exact local quantiles. Mean and weighted group summaries come from
summed numerators and denominators. An any-missing flag can alternatively be
obtained by summing integer flags and testing `> 0`, rather than implementing a
separate global boolean operator. Empty contributions use zero for sums,
positive infinity for minima, negative infinity for maxima, and an empty set for
union, with explicit valid counts distinguishing real infinity from no data.
The existing aggregator sometimes converts infinity sentinels to NaN; record
that behavior during migration rather than quietly changing it.

## Semantics that need decisions before implementation

1. **Missing inputs and typed categories.** Current FETA rejects missing inputs,
   loads numerical float64 or string-label tensors and supports boolean masks.
   String-label initialization and scalar operand transport are implemented;
   dictionary encoding and missing-category policies remain open. Dataset membership is required in addition to initial x/y.
1. **Row alignment after filtering.** Implemented masks must belong to the same context and
   source rows. Applying the same mask to paired tensors must produce compatible
   lineage; independently filtered tensors cannot be multiplied just because
   their shapes match. Retain workers with empty filtered subsets as neutral
   contributors after initialization rather than changing the eligible population.
1. **Result placement and reduction meaning.** Absence of a partitioned axis
   does not imply sum for future kernels. Extrema require min/max; category
   candidates require union; exact local quantiles and dataset summaries require
   an explicitly local, protected result path. Operation semantics must determine
   the reduction. Do not sum local medians/stds, or average worker means.
1. **Fixed categories and edges.** Every worker needs the same ordered category
   dictionaries and numerical bin edges before contributing a matrix/vector.
   Unknown levels, metadata extras, NaN categories and column labels have
   different existing rules across algorithms; keep these policies explicit.
1. **Exact local versus estimated global quantiles.** Describe exposes exact
   pandas dataset quartiles and iterative histogram estimates globally. Outliers
   use exact dataset quantiles and median absolute deviation, including the
   factor 1.4826. A single generic global `quantile` cannot replace all three.
1. **Approved local summaries.** Describe and outlier report return per-dataset
   summaries, not only sums. Current FETA has no general local-value retrieval
   API. Introduce a narrowly scoped summary release contract with worker-side
   privacy checks; keeping a quantile in the KV store alone does not enable the
   algorithm's existing output. Labels/category discovery also require review.
1. **Privacy behavior is algorithm-specific.** Describe suppresses a dataset or
   variable below the threshold and excludes its population from global quartiles.
   Histograms suppress counts below the threshold, including zero. Outliers
   expose zero counts, suppress small nonzero counts, and suppress total count and
   percentage if either tail is suppressed; insufficient nonmissing input raises
   an error. Do not replace these with one universal masking function.
1. **Resource and protocol bounds.** Bound category cardinality, bin count,
   design width, intermediate arrays and repeated percentile rounds. Validate
   enum parameters, axis, dtype and lineage before allocation. No Python source,
   formula evaluator, import path, pickled callable, unrestricted query string,
   automatic retry or batch is implied by this catalogue.

## Statistics implementation order

1. Add absolute value and restricted column/vector shape operations. Numerical
   one-sample/paired t-tests and ungrouped moment/SMD computations can reuse the
   existing algebra and sum protocol on complete inputs.
1. Decide missing/boolean/categorical initialization and mask contracts; add
   comparisons, logical masks, row selection, valid counts and category encoding.
   This unlocks independent t-tests and grouped moment computations.
1. Add local extrema with min/max aggregation, ordered category counting and
   contingency kernels, plus typed discovery where needed. This covers one-way
   ANOVA and categorical tests once their input and release rules are settled.
1. Add histogram counts using explicit common edges. Compose grouped histograms,
   percentile refinement and binned Mann–Whitney; do rank/cumulative calculations
   on completed histograms in the controller.
1. Add design construction from encoded factors and stored aligned rows. Reuse
   existing matrix operations and broadcast coefficients for two-way ANOVA.
1. Add internal exact local quantiles and approved dataset-summary release.
   Preserve describe/outlier privacy and distinct local/global statistics.

Testing each implemented kernel should cover empty local contributions,
all-missing data, unequal worker sizes, globally absent levels, one worker lacking
a group, bin-edge inclusion (last bin closed; others half-open), constant values,
integer percentile refinement, filtered paired alignment and memory limits.
Use the existing standalone statistics suites as migration parity references;
new FETA tests should simulate multiple workers and verify the intended reduction
and release behavior. This review does not implement or migrate those algorithms.

## Extension: all remaining federated families

The following review extends the statistics catalogue to every remaining
non-export module under `algorithms/federated`, including training, prediction,
preprocessing, composition, cross-validation and shared infrastructure. The
complete tree contains 57 Python modules: 43 non-`__init__` modules plus 14 package
initializers. Of the latter, `utils/__init__.py` also implements conversion and
an error type; the others provide exports. This scope covers federated cores,
not additional runtime-only algorithms in Exareme3 or Flower.

Preprocessing and model composition are reviewed as future requirements, not
introduced into FETA's currently confirmed no-preprocessing scope.

### Additional local computation families

These extend the proposed primitive catalogue above. Some can be compositions;
specialized kernels are options to review, not approved protocol additions.

| Computation family | Worker-local contract or composition | Consumers and qualifications |
| --- | --- | --- |
| Stable nonlinear functions | `exp`, `log`, `clip`, `sigmoid`, `xlogy`; preserve zero-times-log semantics and explicit clipping bounds | Logistic likelihood/weights, survival risk weights, GLMM probabilities and derivatives, NB prediction; stable sigmoid is preferable to unchecked exponentiation |
| Row-preserving broadcast | Restricted vector-to-column reshape and scalar/feature/class parameter broadcast | `X * weights[:, None]`, centered/scaled PCA, logistic and NB prediction; no expansion of the distributed record dimension |
| Row extrema and index results | `argmin/argmax(x, axis=1) -> integer tensor`; row-normalization with keep-dimension sums or reshape | Cluster assignments and class predictions; row-wise decisions stay distributed, no global argmin is needed |
| Keyed/scatter accumulation | `segment_sum(values, labels, n_labels)` and class/group counts | K-means cluster sums/counts/inertia, class-feature NB moments, confusion matrices; labels are bounded integer codes with fixed output order |
| Pairwise squared distance | `squared_distances(X, centers) -> (local_rows, n_centers)` | K-means; compose row norms and `X @ centers.T` or use a bounded distance kernel; avoid mandatory `(rows, centers, features)` intermediates |
| Categorical gather | `gather_category_parameters(table, codes, unknown_policy)` | Categorical NB likelihood; per-feature vocabularies can have different sizes, so one rectangular 3-D tensor is not a faithful representation |
| Local product | `product(x, axis=feature_axis)` or sequential elementwise multiplication | Gaussian/categorical NB likelihoods; a feature reduction remains row-partitioned. This is not a required global product operator |
| Density/probability evaluation | Gaussian PDF composition or approved class-wise prediction kernel; ordinal cutpoint sigmoid differences | NB and ordinal GLMM inference; keep per-row likelihoods/probabilities on workers |
| Weighted products | `X.T @ (w * X)` and `X.T @ r`, with column-shaped `w/r` | Logistic Hessian/gradient, weighted OLS-like/group summaries and GLMM derivatives; do not construct an `n × n` diagonal matrix |
| Local row permutation | Stable `argsort(key)`, then aligned `take_rows` on every related tensor | Both Cox variants; changes row lineage and order, does not imply a federation-wide sort |
| Local scans and index lookup | Forward/reverse `cumsum`, bounded `searchsorted`, typed row gather | Classical Cox suffix risk sums, stacked Cox risk-set starts; scans over records must remain explicitly local rather than pretending to be globally ordered |
| Row outer moments | Weighted per-row outer products, or block-wise feature-pair products and suffix scans | Classical Cox second-order risk statistics; current implementation has rank-3 arrays, and flattened rank-2 representations still require placement metadata |
| Survival row expansion | Controlled `stack_survival_rows(X, times, events, grid)` producing aligned design/target references | Stacked Cox; variable output row count, repeated observations and interval indicators require a new lineage contract and preallocation bounds |
| Fold membership | Deterministic worker-local KFold membership or index selection using seed, fold and split count | Cross-validation; apply one membership object to data and target, without returning raw indices or observations to the controller |
| Threshold counts | Comparisons/counts at fixed thresholds, or a bounded ROC-count kernel | Binary scorer's 101 default thresholds; output vectors are additive and thresholds must agree across workers |
| Cluster sufficient statistics | Counts, `X.T @ X`, `X.T @ y`, `y.T @ y`, feature/outcome sums and group-size histogram | LMM; may retain each group's private summaries in the worker store for subsequent parameter-dependent evaluations |
| Symmetric packing | `pack_upper_triangle` with a declared matrix dimension; inverse unpack on controller | LMM and binary GLMM compact derivative contributions; transport optimization, not a new mathematical reduction |
| Cluster-local objective/mode kernels | Bounded random-intercept mode optimization, objective/score/Hessian or objective/gradient; optional private warm-state reference | Binary and ordinal GLMM; inner iterations are local and may differ across groups/workers |
| LMM parameter evaluation | Fixed phase-A GLS summary and phase-B score-term kernels over cached cluster statistics | LMM REML optimization; avoid repeated RPCs for each private group and repeated raw-data scanning |

A mathematical `exp(x)` can be expressed through power with a concrete base,
and a square root through power `0.5`. That does not replace stable clipped
exponential/log-likelihood kernels where overflow, underflow or `0 * log(0)`
semantics matter. Any new function needs a fixed allowlist entry or an explicitly
validated composition, not a generic NumPy/SciPy dispatcher.

### Remaining source-to-library mapping

Difficulty below assesses migration to the current FedTensor language, not the
statistical method in isolation. "Small" still includes missing basic primitives;
"structural" means a new placement, lifecycle, input or local-control contract.

| Source module | Local computation and retained state | Combination / controller work | Main difficulty |
| --- | --- | --- | --- |
| [linear_model/ols.py](../../exaflow/algorithms/federated/linear_model/ols.py) | Intercept, Gram/cross-products, outcome moments, broadcast prediction and squared/absolute residuals | Sum; pseudoinverse/rank, coefficients and inference | Small: intercept/absolute value and prediction typing |
| [linear_model/logistic_regression.py](../../exaflow/algorithms/federated/linear_model/logistic_regression.py) | Intercept; `eta`, stable sigmoid, weights, gradient, weighted Gram matrix and `xlogy` likelihood each iteration | Sum; inverse/pseudoinverse, coefficient update and shared convergence loop; at most 50 iterations in current core | Moderate: nonlinear functions, column-vector shapes and intermediate reclamation |
| [linear_model/cox_regression_classical.py](../../exaflow/algorithms/federated/linear_model/cox_regression_classical.py) | Stable aligned sorting; event counts/covariate sums; clipped exponentials; weighted suffix risk moments and event-grid lookup | Numeric event-time union, sum; Breslow score/information, Newton/ridge update and inference | Structural: sorted local scans, rank-3 risk blocks and release of event times |
| [linear_model/cox_regression_stacked.py](../../exaflow/algorithms/federated/linear_model/cox_regression_stacked.py) | Survival validation, aligned sorting, risk-set row expansion and time indicators; logistic fit on expanded tensors | Max time, sum counts and logistic contributions; event-time union for event-time grid; controller grid construction | Structural: expanding and duplicating aligned rows, large memory and model width |
| [decomposition/pca.py](../../exaflow/algorithms/federated/decomposition/pca.py) | Count/sum/squared sum; broadcast means/scales; standardization; standardized Gram; local projection | Sum; covariance and eigendecomposition on controller | Small algebraically: immutable replacement for current in-place arrays; transform output remains private/distributed |
| [cluster/kmeans.py](../../exaflow/algorithms/federated/cluster/kmeans.py) | Finite validation, extrema, nearest-center assignment, segmented sums/counts and cluster inertia; retain local labels | Min/max/sum; seeded random-range centers, convergence and selection over k/multiple initializations | Moderate: row argmin, scatter sums and memory across repeated fits; no raw-row sampling is required by current initialization |
| [naive_bayes/naive_bayes_gaussian.py](../../exaflow/algorithms/federated/naive_bayes/naive_bayes_gaussian.py) | Per-class/per-feature valid counts/sums/squares; local PDF likelihoods, feature products, normalization and argmax | Sum; class moments, variance smoothing and priors on controller | Moderate: class masking, missing counts and prediction broadcast/product; 3-D prediction can be streamed by class |
| [naive_bayes/naive_bayes_categorical.py](../../exaflow/algorithms/federated/naive_bayes/naive_bayes_categorical.py) | Target discovery; encoded per-class category counts; gathered per-feature probabilities, unknown-code neutral factors and feature products | Label union and sum; smoothing, priors and tables on controller | Moderate: category encoding, ragged per-feature tables, gather and private label predictions |
| [mixed_effects/lmm.py](../../exaflow/algorithms/federated/mixed_effects/lmm.py) | Optional square-root weighting; cached cluster moments; group-size histogram; parameter-dependent phase A/B summaries | Max cluster size, sum; optional group union for rough initialization; GLS solve, REML/profile evaluations, global backtracking and inference | Structural: cluster-local summaries and cluster ownership; optimized current path avoids dense observation covariance |
| [mixed_effects/glmm_binary.py](../../exaflow/algorithms/federated/mixed_effects/glmm_binary.py) | Grouped data; private warm random-intercept modes; bounded local Newton solves; Laplace score/Hessian/objective | Group union and sum; packed symmetric matrix unpack, global Newton update and inference | Structural: variable local inner loops and warm-state handling; summaries have different release semantics from modes |
| [mixed_effects/glmm_ordinal.py](../../exaflow/algorithms/federated/mixed_effects/glmm_ordinal.py) | Grouped ordinal outcomes; cutpoint probabilities and first/second/third derivatives; gathered outcome entries; private warm modes and site objective/gradient | Group union and sum; cutpoint parameterization, BFGS/backtracking, repeated finite-difference information and inference | Most complex: local mode optimization, dynamic gathers, state acceptance and many objective evaluations |
| [mixed_effects/common.py](../../exaflow/algorithms/federated/mixed_effects/common.py) | Group selection, weights, random-intercept covariance helpers, products/trace, histograms, packing, sigmoid and logistic derivative/mode helpers | Shared post-aggregation GLS/REML formulas and line-search helpers | Support module: distinguish used optimized paths from optional dense covariance helpers |
| [model_selection/cross_validation/splitter_kfold.py](../../exaflow/algorithms/federated/model_selection/cross_validation/splitter_kfold.py) | Deterministic local KFold/permutation and aligned train/test selection; currently reusable mutable buffers | No independent reduction | Structural: immutable fold references/lineage; splits are local, not a pooled global KFold |
| [model_selection/cross_validation/cross_validator.py](../../exaflow/algorithms/federated/model_selection/cross_validation/cross_validator.py) | Repeated train/transform/predict/score with per-fold data and target | Estimator/scorer reductions; controller collects per-fold metrics | Structural lifecycle: reset fitted preprocessing per training fold, reclaim private fold/model intermediates, retain shared input context |
| [model_selection/cross_validation/scorer_regression.py](../../exaflow/algorithms/federated/model_selection/cross_validation/scorer_regression.py) | Private predictions, residual square/absolute sums, test count/outcome moments | Sum; RMSE, MAE, R² and held-out F diagnostic with training degrees of freedom | Small once prediction/absolute value and aligned held-out tensors exist |
| [model_selection/cross_validation/scorer_classification.py](../../exaflow/algorithms/federated/model_selection/cross_validation/scorer_classification.py) | Probability prediction; 0.5 confusion counts plus TP/FP/TN/FN at shared thresholds | Sum; ROC rates and confusion-derived metrics | Moderate: masks and threshold-count kernel; do not transfer row probabilities |
| [model_selection/cross_validation/scorer_multiclass.py](../../exaflow/algorithms/federated/model_selection/cross_validation/scorer_multiclass.py) | Private predicted labels, label mapping, bounded confusion scatter counts; ignores unknown true/predicted labels | Sum; per-class metrics and fold summary formatting | Moderate: labels, gather/argmax, fixed confusion order and unknown policy |
| [preprocessing/one_hot_encoder.py](../../exaflow/algorithms/federated/preprocessing/one_hot_encoder.py) | Observed nonmissing category discovery; equality indicator columns; numerical passthrough; drop first sorted level | Union; learned category/feature order on controller | Scope extension: preprocessing is currently excluded; fold-local fitting prevents leakage |
| [preprocessing/ordinal_encoder.py](../../exaflow/algorithms/federated/preprocessing/ordinal_encoder.py) | Encode explicit metadata category order; reject unknowns or encode with supplied sentinel | No discovery reduction | Scope extension: typed categorical inputs and unknown/missing policies; result is numeric codes |
| [preprocessing/passthrough.py](../../exaflow/algorithms/federated/preprocessing/passthrough.py) | Ordered numeric column selection/stacking, zero-fill absent columns, empty-column shape | None | Small mechanics, but preserve explicit missing-column policy rather than silently broadening input validation |
| [compose/column_transformer.py](../../exaflow/algorithms/federated/compose/column_transformer.py) | Select/encode blocks and concatenate aligned columns | Transformer reductions; feature-label/remainder resolution on controller | Scope extension: resolve selections from trusted metadata; callable selectors never cross the RPC boundary |
| [pipeline.py](../../exaflow/algorithms/federated/pipeline.py) | Chains transforms and estimator fit; no independent mathematical kernel | Delegate reductions; controller coordinates typed stages | Structural API change: current interfaces expect DataFrames/ndarrays, not FedTensor; do not serialize pipeline instances/callables |
| [sql/sql.py](../../exaflow/algorithms/federated/sql/sql.py) | Local filter/group/column extraction and per-group statistics | Group-key union and delegated reductions | Typed translation needed: arbitrary DataFrame query strings and aggregation callables are outside the allowlist |
| [utils/aggregators/numpy_aggregator.py](../../exaflow/algorithms/federated/utils/aggregators/numpy_aggregator.py) | Missing masks/replacement, valid counts and local reductions | Sum/min/max/union and ratios | Preserve all-missing and sentinel semantics; no additional average operator is required |
| [utils/agg_client.py](../../exaflow/algorithms/federated/utils/agg_client.py) | Protocol declarations only | Declares sum/min/max/union | No kernel; legacy protocol declaration does not extend FETA |
| [utils/interfaces.py](../../exaflow/algorithms/federated/utils/interfaces.py) | Estimator/result/splitter/scorer/transformer protocols | None | Replace ndarray-oriented signatures with appropriate typed distributed interfaces when migrating |
| [utils/__init__.py](../../exaflow/algorithms/federated/utils/__init__.py) | Arrow/array conversion and BadInputError | None | Loading boundary only; algorithms must not cast FedTensor to NumPy |

All other package initializers export classes/functions and add no worker
computation kernels. The OLS and SQL entries are repeated as first-class family
members here because the statistics review previously only treated them as
transitive dependencies.

### Global operations after the full review

The necessary operator set is still **sum, min, max and deterministic union**.
Union now covers numeric event-time grids and cluster identifiers as well as
categories; its cardinality, typed encoding and permitted release need separate
contracts. There is still no requirement for a global product, global argmin,
global scan, raw-data gather or separate average operator. NB feature products
and K-means nearest-center argmin are local, while norms, eigendecompositions,
Cholesky/linear solves, BFGS updates and inference on completed global matrices
belong to ordinary controller NumPy/SciPy.

Preserving those boundaries matters: a scan over sorted worker-local times is
not a scan over the pooled federation; a local class probability is not an
approved global result; a local random-effect mode is not an additive partial.

### Hard problems for the FedTensor language

These are concrete migration risks identified from the source. Proposed remedies
below are not changes to accepted FETA scope or permissions.

1. **Automatic aggregation cannot represent private local reductions.**
   Today `Runtime.apply` interprets removal of the partitioned axis as global sum.
   A GLMM random-intercept mode needs a sum within one private cluster, checks
   convergence locally and repeats a different number of times for each cluster.
   A controller-side sequence of ordinary `.sum()` calls would aggregate different
   clusters/workers together. Generic local control flow would also tempt callers
   to obtain private scalars. Prefer bounded fixed operations for cluster-mode
   and derivative evaluation, with private state references; otherwise design a
   genuinely separate local-reduction contract before expressing this in algebra.
1. **No distributed object represents worker-specific replicated-axis tables.**
   LMM group moments, risk-grid blocks, histogram counts and derivative vectors
   are partial summaries indexed by groups/features/times, not partitioned rows.
   Some should be immediately reduced; others must remain private to be reused.
   `partition_axis = -1` alone cannot distinguish those roles. Operations need
   explicit semantic output placement and reduction rules inside the runtime.
   This does not imply introducing the rejected developer-facing SharedTensor.
1. **Classical Cox requires order and rank-3 risk moments.**
   The source stores `(local_rows, p, p)` weighted outer moments and suffix arrays,
   then gathers `(event_times, p, p)` risk summaries. FETA rejects rank above two.
   A fixed risk-summary kernel may flatten moments as `(event_times, p*p)` with
   bounded metadata, or compute feature pairs/blocks without 3-D storage. Merely
   flattening shape does not fix placement, scan semantics or memory. Repeating
   comparisons and products through RPC for every time/feature pair is possible
   mathematically but costly; no global scan is necessary for current Cox.
1. **Stacked Cox changes what a record is and can greatly enlarge the data.**
   It repeats each observation in each eligible risk set and adds time indicators.
   Expansion can approach `n_local * n_grid` rows and `p + n_grid` columns, before
   iterative logistic intermediates. Original and expanded tensors must have
   different row lineage; expanded target and design must share that lineage.
   Precompute allocation bounds. A streaming/block summary formulation is a
   substantial alternative requiring parity validation, not a reshape operation.
1. **Mixed-effects clusters must have an ownership contract.**
   The reviewed LMM/GLMM computations form groups locally. Existing mixed-effects
   tests partition complete centers to workers. If one logical cluster spans
   workers, adding locally optimized cluster likelihoods/modes is not equivalent
   to fitting one pooled random intercept. A category union only counts labels;
   it does not repair this. Require whole-cluster locality or design cluster-wise
   distributed sufficient-statistic/mode coordination. Do not claim arbitrary
   row partitioning supports these models.
1. **Immutable results accumulate across iterations.**
   The current store adds every operation result to the context byte count and
   removes them only when the context is cleaned up. Logistic/K-means/CV can
   retain many obsolete row-sized arrays and duplicate global parameters, hitting
   the 256 MiB context cap even when one iteration fits. Add safe reference
   release, operation scopes or a carefully bounded lifetime policy that retains
   explicitly reused results. Preserve cancellation/expiry publication checks;
   do not solve this by silently increasing limits or mutating inputs.
1. **Warm modes and rejected optimizer trials need private state versions.**
   GLMM objectives return a next warm-mode array alongside additive contributions.
   Ordinal backtracking evaluates trial states, and observed information evaluates
   two perturbed scores per parameter. Warm state must remain worker-local and
   only advance according to accepted-trial semantics; candidate/rejected states
   need cleanup. One request can still mean one fixed kernel, but heterogeneous
   private/global outputs require a defined response/storage contract. They must
   not be blindly concatenated and summed.
1. **Global loops are easy; inner loops and RPC volume are not.**
   Logistic Newton updates, K-means center updates and LMM/ordinal backtracking
   can run as ordinary controller Python using global results. However every
   immediate reduction is a separate aggregation step, and each primitive is a
   worker request. Ordinal finite-difference inference adds `2*q` site-gradient
   evaluations, each with nested cluster-mode work. Measure requests, rounds and
   peak memory. A fixed bounded domain kernel is compatible with one operation
   per request; automatically batching operations remains an unaccepted change.
1. **Rank-3 broadcast need not become a universal requirement.**
   K-means currently computes `(rows, centers, features)` differences; Gaussian
   NB predicts with `(rows, classes, features)` PDFs. Both can be expressed using
   2-D distance matrices or class-wise streamed computations. Row shape utilities,
   feature-product reduction, safe gather and argmin/argmax are still missing.
   Algebraic distance expansion can change floating-point cancellation/tie
   behavior; switching NB to log probabilities can change underflow behavior.
   Treat either rewrite as an explicit parity tradeoff, not an automatic fix.
1. **Fold/permutation lineage and model lifecycle are beyond shape typing.**
   Train/test tensor pairs need the same private membership/permutation ID; folds
   must not mix with other folds or independently sorted tensors. Existing KFold
   is worker-local and rejects a worker with fewer observations than splits.
   FETA's initialization skip rule does not define fold-time exclusion. Fit
   category discovery/transformers on training data only, decide state lifetime
   across folds and validate held-out pairing. Adding a generic index method
   without these contracts creates leakage or wrong aggregates.
1. **Predictions and discovery have distinct release boundaries.**
   Model coefficients/components/centers are completed global model outputs;
   predictions, class probabilities, cluster labels, sorted times and fold indices
   are private row-level tensors. Scorers should reduce counts/moments locally.
   Exact event-time union and cluster/category identifiers can themselves reveal
   information; reuse the existing policy only where allowed, and bound/authorize
   each new FETA release instead of exposing arrays for convenience.
1. **Input/API scope is broader than x/y numerical tensors.**
   Survival needs time/event columns; mixed effects require center IDs and optional
   weights; categorical learners need encodings/vocabularies; preprocessing needs
   column metadata and trained state. Prepared y can hold two survival columns,
   but center/weight inputs still need a deliberate additional-input contract.
   Feature labels remain model/schema metadata, not named tensor axes. The
   ndarray/DataFrame estimator, pipeline and scorer APIs cannot simply accept
   FedTensor while their bodies call `np.asarray` and arbitrary selectors.

### Source behaviors to audit during migration

These observations are not fixes made by this review:

- PCA's default `copy=False` mutates local arrays; `fit_transform` calls
  `transform` after `fit`. FedTensor's immutable expressions should not emulate
  unintended repeated mutation without settling the intended parity contract.
- Gaussian NB uses the first feature's nonmissing count as its class count.
  Per-feature missingness can make that differ from true class row count. Preserve
  or explicitly revise this behavior instead of silently substituting `len`.
- Categorical NB skips target union and some reductions for locally empty y.
  A direct port can produce divergent collective schedules. Establish shared
  dictionaries/steps centrally and keep neutral contributions where required.
- Mixed-effects common helpers include dense `n_j × n_j` inverse covariance
  matrices, but the active LMM path already uses cached cluster moments and
  scalar random-intercept formulas. Migrating the optimized path avoids adding
  general private matrix inversion as a misleading prerequisite.
- Pipeline/column-transformer support callable selectors and ndarray/DataFrame
  conversion. These are trusted local Python APIs today, not safe RPC payloads.

### Roadmap across all families

1. Numerical algebra path: PCA and OLS, then logistic regression with stable
   nonlinear operations, typed row/column shape rules and intermediate release.
1. Categories and row decisions: shared encoding, masks/gather, segmented moments,
   class/cluster count matrices, row argmin/argmax. Cover NB, K-means and scorers
   alongside the existing statistics expansion.
1. Lifecycle/composition: typed transform/model interfaces, aligned fold views,
   training-only discovery and bounded per-fold/per-iteration storage. Keep the
   confirmed no-preprocessing first version until its scope is explicitly revised.
1. Survival: decide numeric time-grid release and risk/expansion placement; choose
   bounded classical-risk and stacked-design kernels or validate block alternatives.
1. Mixed effects: settle whole-cluster ownership, private cached group summaries,
   mode-state acceptance and bounded local optimizer kernels. Migrate LMM before
   binary GLMM, then ordinal GLMM's nested optimizer and finite-difference inference.

In addition to statistics parity cases, validate seeded K-means starts/ties,
constant PCA features, logistic extreme probabilities, NB unknowns/underflow,
Cox tied events/censoring/suffix boundaries, whole-cluster partitioning, rejected
warm-state trials, local fold reproducibility, training/test separation and peak
store growth. Documentation-only review: no new kernels, dtype support,
preprocessing, privacy release or runtime routing is enabled here.
