# Federated package input validation

Validation is local and raises `BadInputError`, a `ValueError` subclass. Expected
conversion failures are reported without echoing row values. The package does
not coordinate error propagation or cancellation: callers own the distributed
execution lifecycle. Exareme3 converts worker input errors to user errors and
cleans up the request's aggregation context; running peers need not stop
immediately.

The changes preserve successful numerical calculations, public signatures,
result shapes, defaults, existing conversions, missing-value policies, and
numerical fallbacks. Existing production and algorithm tests and fixtures are
unchanged. New tests belong in `tests/standalone_tests/federated_validation/`.

## Shared checks

The implementation is in `federated/validation/__init__.py`. It reuses public
scikit-learn `check_array`, `check_consistent_length`, and `check_scalar` helpers.
NumPy conversion is shared with the existing `utils.to_numpy` adapter and
re-exported through `validation`, so its conversion/error logic has one owner.
The package does not inherit the sklearn estimator API or import its private
validation APIs. The package retains its results-oriented estimator interfaces.

Array validation defaults to the existing floating-point conversion. Callers can
specify a dtype when the algorithm has already normalized an array; mixed-effects
validation preserves integer outcome arrays instead of creating a discarded float
copy. Finite-value requirements remain unchanged. Ordinary NumPy-backed pandas
Series use `to_numpy(dtype=..., copy=False)`; extension-backed Series retain the
NumPy conversion path and its missing-value behaviour.

Category-order validation returns a validated pandas categorical dtype. The
ordinal encoder reuses it when constructing categorical values, avoiding duplicate
category validation within a transformation. Each transformation still validates
the current category order, including changes made after fitting; categories are
never inferred from the data.

| Helper | Responsibility |
| --- | --- |
| `ArrayPolicy`, `check_array` | Numeric conversion, explicit rank and finite-value policy; no universal sample/feature minimum. |
| `as_numeric_array` | Existing NumPy conversion, including explicit integer encoding where already used. |
| `as_scalar` | Existing `int`/`float` conversion with safe input errors; validation does not introduce new coercions. |
| `ParameterRule`, `BOOLEAN_RULE`, `validate_parameters` | Type checks on current parameter values without coercion. |
| `check_scalar`, `check_parameter_ranges` | Numeric types, finiteness and method-specific open/closed bounds. Unbounded mathematical limits are explicitly allowed where supported. |
| `check_choice` | Supported method options without echoing unsupported supplied values. |
| `check_dimensions` | Rank checks independent of dtype, normalization and missing-value handling. |
| `check_consistent_length`, `check_classification_target` | Matching observation counts and 1D class-label structure without changing label dtype; prevents accidental broadcasting. |
| `check_feature_schema`, `check_feature_names` | Effective prediction width and supplied feature-name length. No invented feature-order discovery or uniqueness restriction. Empty optional names retain the existing fallback. |
| `check_sample_counts` | Existing or necessary method-specific sample thresholds, applied to counts already aggregated by the algorithm. |
| `check_dataframe` | DataFrame type and required column access, without imposing unique columns or dropping rows. |
| `check_interval` | Ordered limits, with explicit support for equal histogram/clipping bounds or unbounded clipping limits. |
| `check_fitted` | Required fitted attributes before PCA transformation. |
| `check_named_steps` | Component tuple structure and unique, nonempty names before fitting components. |
| `check_category_order` | pandas' existing category-order constraints, without learning or sorting categories. |
| `check_survival_target`, `check_survival_inputs` | Shared Cox target parsing and the two models' existing local survival requirements. |

`ValidationIssue` supplies a code, input name and safe message on structured
helper errors. Existing method-specific `BadInputError` checks remain valid;
callers should catch `BadInputError` rather than depend on its private subclass.
Transport and programming errors are not captured by a validation context or
converted into validation failures.

## Algorithm coverage

Checks run on effective inputs after each method's existing conversions.
Constructor conversions/checks remain at construction where already present;
operations also check current mutable parameters that they consume.

| Algorithm or entry point | Validation and retained behavior |
| --- | --- |
| OLS | Boolean intercept setting; finite numeric features/outcome and matching rows after existing flattening/intercept addition. Prediction width; confidence level in `[0, 1]`. Retains singular-matrix pseudo-inverse, zero-feature/empty-result paths and existing residual-degree fallbacks. No new `n > p` restriction. |
| Logistic regression | Boolean intercept; finite numeric features and encoded binary targets; matching rows after existing flattening, intercept and transposition. Existing global observation/class-count thresholds remain. Prediction width; probabilities and IRLS unchanged. |
| Classical Cox | Shared finite `[time, event]` target, positive times, binary events, matching rows and existing nonempty local observations/covariates. `ties='breslow'`; positive iteration/tolerance/step settings and nonnegative ridge. Effective iteration count must be integral. Optional feature names and prediction width. |
| Stacked Cox | Same survival contract. Supported grid strategy and positive bin count; existing time-bin ordering, origin/coverage and event-time checks retained, with explicit non-finite bin rejection. Flat prediction input remains one observation. Existing event-count and identifiability guards remain. |
| PCA | Finite 2D fitting features and Boolean copy setting; at least two global observations from existing aggregated moments. Transformation requires fitted state and matching width. Constant-feature scaling, copying, writable/read-only handling and mutation order remain unchanged. |
| Gaussian Naive Bayes | Existing numeric feature conversion/flat reshaping; finite 2D effective features matching `x_vars`; 1D aligned labels; nonnegative finite smoothing. Empty/local-class behavior and preset class selection remain. Prediction uses the same feature checks. |
| Categorical Naive Bayes | Shared training/prediction encoding; feature rank/width and aligned 1D targets. Retains NaN-to-unknown conversion, integer truncation and unknown-code handling. Existing empty-partition collective branching is unchanged, rather than repaired as part of validation. |
| K-means and K selection | Effective positive integer cluster/iteration/initialization counts; finite nonnegative tolerance and valid random seed; existing initialization choices and K-range checks. Feature rank/name checks and existing center/feature checks remain. No change to initialization, empty-feature behavior or existing finite-status collective. |
| LMM | Shared finite feature/outcome, identifier rank/length, optional finite nonnegative weights. Existing variance bounds, initial variances and iteration relationships retained; checks apply to current solver settings and Boolean flags. Unbounded upper variance limits remain supported. Prediction width. |
| Binary GLMM | Same shared mixed-effects checks; existing encoded binary outcome rule and required Laplace correction. Positive finite solver/mode settings, effective integer iteration counts, nonnegative ridge, valid clipping interval and initial log variance. Existing Boolean casts, disabled clipping (`None`), equal and unbounded clipping limits remain supported. Prediction width. |
| Ordinal GLMM | Shared mixed-effects checks; existing integer target conversion and range `[0, K-1]`; effective integral `K >= 2`; solver/mode/clipping checks as applicable. No change to ordinal parameterization, cutpoints, class availability or numerical stabilization. Prediction width. |
| One-way ANOVA | Group/category alignment and confidence level; at least two supplied and globally nonempty groups. Positive residual degrees of freedom checked using existing aggregated counts. Existing per-group flattening and empty local groups retained. |
| Two-way ANOVA | Required DataFrame columns; supported effective sums-of-squares type (`1` or `2`); existing factor-level guards. Numerical model validation delegates to OLS. No new full-rank or balanced-design restriction. |
| One-sample t-test | Finite hypothesized mean, confidence level in `[0, 1]`, and supported alternative. Existing flattening, missing-value propagation, zero-variance behavior and global count guard retained. |
| Paired t-test | Same confidence/alternative checks; matching flattened sample lengths. Existing global count guard and numerical behavior retained. |
| Independent t-test | Required group/value columns; confidence/alternative checks. Existing group selection, missing-value propagation and global group/count guards retained. |
| Pearson correlation | Required columns and confidence level in `[0, 1]`; existing global sample guards retained. No new four-observation CI restriction, nonconstant-feature restriction or blanket missing-value rejection. |
| Simple histogram | 1D input, positive integral bin count, finite ordered explicit bounds (equal bounds allowed). Existing NaN filtering, empty local arrays, global-range guard and NumPy binning retained. |
| Wilkinson histogram | 1D input, positive integral bin hint and finite nonnegative step floor. Existing snapped binning, constant ranges and integer floors retained. |
| Categorical histogram | Shared 1D check; category/NA discovery and counts unchanged. No numeric conversion or finite-value restriction. |
| Grouped histogram | Required frame columns; numerical branch delegates bin checks and validates its strategy. Existing discovery, grouping and privacy masking retained. |
| Percentile/quartiles | `q` in `[0, 1]`, positive integral bins and Boolean integer-mode flag. Existing histogram refinements, NaN behavior, empty/all-missing `None` result and integer-specific computation retained. |
| Binned Mann–Whitney U | Supported alternative, Boolean continuity flag and integral bins of at least two, before aggregation. Existing filtering of non-finite observations, group counts, ties and zero-variance guard retained. |
| Cross-tabulation | Required frame columns and Boolean `dropna`; existing category union, explicit-category reindexing, missing-value handling, pruning and empty-table guard retained. |
| Chi-squared | Delegates frame/table validation to cross-tabulation and mathematical table checks to SciPy; no duplicated wrapper validation or new minimum category count. |
| Fisher exact | Same cross-tabulation validation and existing final 2x2 requirement. No extra category-discovery collective added. |
| Describe | Frame check; existing per-variable missing-value handling, missing-column behavior and local/global record privacy masking retained. |
| Descriptive statistics facade | Delegates to describe, histogram and Pearson implementations, without duplicating their checks. |
| Primitive statistics | Finite numeric `ddof` where consumed, including existing support for negative/fractional finite values; existing ndarray/rank, paired-shape and global-count guards retained. No new blanket missing-value rule. |
| Standardized mean difference | Existing sample type/rank and global group-count checks retained; grouped entry checks required columns and effective minimum-group-size setting before SQL execution. |
| One-hot encoder | Frame type and safe numeric-column conversion. Missing columns still become zeros; duplicate-column first-selection behavior, category discovery/order and reference-category omission remain. No new aggregation capability required. |
| Ordinal encoder | Supported unknown-category policy, existing explicit category order, safe existing integer conversion and frame checks. Category-order constraints are checked before encoding. Unknown/missing columns and ignored categories retain existing sentinel behavior; no aggregation added. |
| Passthrough | Frame type and safe existing numeric conversion; missing columns and empty selections retain zero/empty output behavior. |
| Column transformer | Remainder option, named component structure and frame type before fitting; output rank/row alignment before concatenation. Existing 1D or 2D component outputs, column selection, missing-column behavior and zero-column output retained. |
| Pipeline | Frame, component structure/names and target row alignment before fitting; existing component interface checks retained. Empty pipeline fitting and output contracts unchanged. |
| K-fold splitter | Existing integer/bool conversions; effective integral split count of at least two, observation requirements, row alignment and public sklearn KFold configuration checks. Shared fold setup preserves buffer reuse and splitting order. |
| Cross-validator | Frame/array outcome row alignment before splitting; model/splitter/scorer checks delegate to those components. |
| Regression scorer | Prediction/target length alignment before subtraction and nonnegative integer training/feature counts; existing empty-test early path and global score guards retained. No extra aggregation or changed score formula. |
| Binary classification scorer | Threshold rank, encoded binary target and prediction/target alignment. Threshold order, empty thresholds and infinite threshold sentinels remain accepted. |
| Multiclass scorer | Nonempty prediction/target rank and length alignment; existing empty paths and ignored unknown labels retained. |
| Outlier reporting/winsorization | Supported strategy/tail, finite valid fold and required report columns. Existing numerical coercion, per-dataset computation, suppression and privacy thresholds remain. Direct core callers receive input errors rather than relying on wrapper parameter validation. |
| SQL and aggregation utilities | Existing specialized array/category/aggregation checks retained. No blanket numeric conversion of categorical values and no new collectives. |

## Exareme3 boundary

Redundant mathematical checks are removed from:

- One-way and two-way ANOVA: category/level counts are checked in the core.
- Fisher exact: the core enforces the final contingency-table shape.
- Gaussian and categorical Naive Bayes CV: the shared splitter enforces local
  row/split requirements.

Category discovery and feature construction remain in the wrappers where needed.
Request shape, variable selection, metadata, category interpretation,
authorization and privacy validation remain at their existing runtime boundaries.
Outlier request validation continues to call the core fold helper while retaining
metadata and user-request checks. Cox event-variable selection and runtime
category-to-binary conversion remain request/metadata concerns.

## Invalid-input errors and compatibility

New or clearer `BadInputError` failures include malformed numeric inputs,
non-finite solver settings, unsupported options, mismatched target or prediction
lengths, invalid effective feature widths, bad component structure, invalid
category ordering and insufficient global PCA/ANOVA observations. These replace
incidental broadcasting, NumPy/pandas/SciPy failures or invalid mathematical
results for unsupported inputs. Structured errors do not include raw row values.
Existing specialized primitive/type errors are retained where no migration was
needed. Global count checks use existing aggregates, never an extra validation
round.

There are no added aggregation calls. The existing K-means finite-status sum is
retained. Successful mathematical collective calls and their order are unchanged;
invalid inputs can fail earlier. Existing missing-value, constant-data,
singular-matrix and empty-result paths are intentionally method-specific.

No API schema, deployment, dependency, authorization or privacy threshold changes
are required. To roll back, revert the validation integrations, shared helpers,
wrapper guard removals and their documentation together; no data migration or
regeneration of expected numerical fixtures is needed.
