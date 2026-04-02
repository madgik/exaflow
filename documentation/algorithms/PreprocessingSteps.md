# Overview

This module provides **federated preprocessing primitives** for statistical analysis
and modeling within the MIP framework, designed for privacy-preserving,
distributed (federated) data environments such as EBRAINS.

The goal is to support **standard preprocessing operations** (scaling, normalization,
transformation, imputation, outlier handling) **without sharing raw data** and
using only **allowed federated aggregation primitives** (e.g. sum, min, max, count,
histograms, union of categories).

These preprocessing components are **model-agnostic** and can be reused by any
statistical function or model implemented in MIP.

______________________________________________________________________

## Federated Preprocessing in MIP

### 1. Standard Order of Preprocessing

In statistical modeling and machine learning, preprocessing follows a well-established order.
This order is important because each step changes the data, and changing the order changes the result.

**Standard preprocessing order**

For each variable (column), preprocessing follows this sequence:

1. Missingness handling
1. Outlier handling / clipping
1. Power or distribution transformation
1. Scaling / normalization

Not all variables need all steps.
A pipeline may stop earlier depending on the variable type and model.
For categorical variables, the typical branch is:
Missingness handling → Categorical encoding (e.g. OneHotEncoder).

**Why this order?**

| Step | Why it comes here |
| ----------------------- | -------------------------------------------------------------------- |
| Missingness | Missing values must be resolved before any numeric operation |
| Outliers | Extreme values distort transformations and scaling |
| Distribution transforms | These assume “clean” values |
| Scaling | Should be the last step so coefficients and distances are comparable |

______________________________________________________________________

### 2. Classification of Implemented Preprocessing Classes

Below we classify each implemented class according to where it belongs in the preprocessing order.

#### A. Missing Data Handling (FIRST)

These must always run before anything else.

**SimpleImputer**
Category: Missing data handling
Purpose: Replace missing values using federated statistics.
Strategies:

- mean → global mean

- median → histogram-based federated median

- constant → user-defined value

- mode

  - numeric → histogram peak
  - categorical → global_union + global_sum(counts)

This is the core missing-data operator.

**MissingIndicator**
Category: Missing data handling (augmentation)
Purpose: Add binary flags indicating whether a value was missing.
Used:

- before or alongside imputation
- especially important in clinical data

This does not replace imputation — it adds information.

A variable may use at most one imputation strategy, optionally combined with a MissingIndicator.
Missing handling:

- max 1 Imputer
- MissingIndicator optional

**Example how to use**

For each selected variable (column) x, MissingIndicator can create an extra binary feature column that marks whether the original value was missing.

Original column: x

New indicator column: x\_\_missing (naming can be whatever you standardize)

Suppose a hospital has a numeric covariate age:
| patient_id | age |
| ---------: | --: |
| 1 | 63 |
| 2 | NaN |
| 3 | 51 |

After MissingIndicator + FederatedSimpleImputer(strategy="median"), the local design matrix becomes:
| patient_id | age_imputed | age\_\_missing |
| ---------: | ----------: | -----------: |
| 1 | 63 | 0 |
| 2 | 58 | 1 |
| 3 | 51 | 0 |

age\_\_missing is the new column created by MissingIndicator.

age_imputed is the original column after imputation (here 58 is the federated median).

Why this matters in our environment

In clinical data, “missing” is often informative (not random). If we only impute, we may hide that signal. With MissingIndicator, models like logistic/Cox/mixed-effects can learn patterns like:

“If age is missing, outcome risk changes”

“If a lab test wasn’t measured, that itself is informative”

______________________________________________________________________

#### B. Outlier Handling / Clipping (SECOND)

Applied after missing values are handled, before transformations.

**Winsorizer**
Category: Outlier handling
Purpose: Clip extreme values to quantiles
Example: [1%, 99%] or [5%, 95%]
Uses: histogram-based federated quantiles

**OutlierDetector**
Category: Outlier diagnostics (non-destructive)
Modes:

- "zscore" → mean / std
- "robust" → median / IQR
  Output:
- boolean mask identifying outliers

Detects outliers but does not modify data by default.

At most one clipping operation per variable. Outlier detection is optional and non-destructive. OutlierDetector → Winsorizer is recommended.
Outliers:

- max 1 Winsorizer
- OutlierDetector optional (diagnostic)

______________________________________________________________________

#### C. Power / Distribution Transforms (THIRD)

Used to make distributions more symmetric or normal-like.
At most ONE of these per variable.

**YeoJohnsonTransformer**
Category: Power transform
Purpose: Normalize skewed distributions
Works with:

- positive,
- zero,
- negative values.
  Uses:
- federated MLE-like λ selection
- histogram-safe aggregation

Most general power transform.

**Log1pTransformer**
Category: Power transform
Purpose: Handle strong right skew
Formula:
\[
x' = \\log(1 + x)
\]
Constraints:

- requires non-negative data (unless signed mode is used)

**QuantileTransformerApprox**
Category: Distribution transform
Maps data to:

- Uniform(0,1) or
- Normal(0,1)
  Uses:
- federated histograms
- approximate CDF

Strong transform — changes interpretation.

Exactly zero or one power / distribution transform per variable.

______________________________________________________________________

#### D. Scaling / Normalization (LAST)

Scaling must be the final step.

**StandardScaler**
Category: Scaling
Z-score normalization:
\[
x' = \\frac{x - \\mu}{\\sigma}
\]
Used by:

- linear regression
- logistic regression
- Cox regression
- mixed-effects models
- KMeans

**RobustScalerApprox**
Category: Scaling (robust)
Uses:

- median
- IQR

Preferred when outliers are present.

**MinMaxScaler**
Category: Scaling (range-based)
Maps values to:
\[
[0, 1] \\text{ (or other range)}
\]
Mainly for:

- distance-based models
- visualization
- bounded pipelines

A variable may use at most one scaling method, and scaling must be the final step.

______________________________________________________________________

#### E. Categorical Encoding (AFTER MISSING HANDLING)

Applied to categorical variables after imputation and before model fitting.

**OneHotEncoder**
Category: Categorical encoding
Purpose: Convert each category into a binary indicator column
Uses:

- federated global category union
- local deterministic binary expansion

Prevents artificial ordinal assumptions for nominal variables.

**OrdinalEncoder**
Category: Categorical encoding (ordered)
Purpose: Convert ordered categories into integer ranks
Uses:

- user-defined ordered categories, or
- federated global category union + deterministic ordering

Preserves ordinal relationships for ordered categorical variables.

For each categorical variable, use at most one encoding strategy.

______________________________________________________________________

### 3. Classification per Implemented Preprocessing Functions

| Preprocessing Functions | Applicable To (Algorithms) | Variable Type | Role / Notes | Typical Use | Typical Position in Pipeline |
| -------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| **SimpleImputer (mean)** | Linear, Logistic, Cox, PropensityScore, Mixed-Effects | Numerical (continuous / discrete) | Replaces missing values using global mean; assumes roughly symmetric distribution | Fill missing numeric values using global mean | 1st – Missing Data Handling |
| **SimpleImputer (median)** | Linear, Logistic, Cox, PropensityScore, Mixed-Effects | Numerical (continuous / discrete) | Robust to outliers; preferred for biomedical data | Robust missing-value imputation for skewed data | 1st – Missing Data Handling |
| **SimpleImputer (mode – categorical)** | Logistic, Ordinal Logistic, PropensityScore, Mixed-Effects | Categorical (nominal / ordinal) | Preserves category set using federated counts | Fill missing categories using global mode | 1st – Missing Data Handling |
| **OneHotEncoder** | Logistic, Linear (with dummy vars), PropensityScore, Mixed-Effects | Categorical (nominal) | Expands each category into binary columns; avoids false ordinal structure | Encode nominal predictors to binary indicators | After Missing Handling (before model/scaling) |
| **OrdinalEncoder** | Ordinal Logistic, Linear, Tree-based, Mixed-Effects | Categorical (ordinal) | Maps ordered categories to integer ranks | Encode ordered predictors with preserved ranking | After Missing Handling (before model/scaling) |
| **SimpleImputer (constant)** | All models using predictors | All types | Fills missing with user-defined constant (e.g. 0, “unknown”) | Replace missing values with fixed constant | 1st – Missing Data Handling |
| **MissingIndicator** | Linear, Logistic, Cox, PropensityScore, Mixed-Effects | All types | Adds binary feature indicating missingness; does **not** replace imputation | Preserve information about missingness (clinical relevance) | 1st – Missing Data Handling (optional augmentation) |
| **Winsorizer / Clipper** | Linear, Logistic, Cox, Mixed-Effects | Numerical (continuous / discrete) | Clips extreme values to quantile bounds; improves stability | Limit influence of extreme outliers using quantiles | 2nd – Outlier Handling |
| **OutlierDetector** | All models (diagnostic only) | Numerical | Detects outliers (z-score / robust); does not modify data | Identify extreme values (diagnostic, non-destructive) | 2nd – Outlier Handling (optional) |
| **Yeo–Johnson Transformer** | Linear, Logistic, Cox, Mixed-Effects | Numerical (continuous / discrete) | Power transform for skewed data; works with negatives | General power transform for skewed distributions | 3rd – Power / Distribution Transform |
| **log1p Transformer** | Linear, Logistic, Cox | Numerical (non-negative discrete / continuous) | Log-like transform for counts, costs, biomarkers | Reduce strong right skew in count-like data | 3rd – Power / Distribution Transform |
| **asinh Transformer** | Linear, Logistic, Cox, Mixed-Effects | Numerical (continuous / discrete) | Log-like transform that supports zeros and negatives | Log-like transform supporting zeros and negatives | 3rd – Power / Distribution Transform |
| **QuantileTransformerApprox** | Linear, Logistic, KMeans | Numerical (continuous) | Maps distribution to uniform or normal; strong transformation | Map data to uniform or normal distribution | 3rd – Distribution Transform |
| **StandardScaler (z-score)** | Linear, Logistic, Cox, Mixed-Effects, KMeans | Numerical | Centers to mean 0, std 1; mandatory for KMeans | Standardize scale (mean 0, std 1) | 4th – Scaling / Normalization |
| **RobustScalerApprox** | Linear, Logistic, Cox, Mixed-Effects | Numerical | Uses median/IQR; preferred with outliers | Robust scaling using median and IQR | 4th – Scaling / Normalization |
| **MinMaxScaler** | KMeans, visualization pipelines | Numerical | Maps to fixed range; distance-based or bounded pipelines | Map values to fixed range (e.g. [0,1]) | 4th – Scaling / Normalization |
| **(Optional) Scaling** | PearsonCorrelation, Covariance | Numerical | Scaling is optional; changes interpretation (especially covariance magnitude) | Optional normalization for interpretability | Optional – After Missing Handling |
| **Missing Handling Only** | MannWhitneyUTest | Numerical / Ordinal | Missing handling allowed; power transforms and scaling are discouraged unless justified | Preserve validity of non-parametric test | 1st – Missing Data Handling only |

## Metadata Update Matrix

| Preprocessing Step | min / max | isCategorical | sql_type | Why |
| -------------------------------------------- | --------------------------- | ------------- | -----------------------| ------------------------------------------- |
| **No preprocessing** | ❌ unchanged | ❌ unchanged | ❌ unchanged | Raw data preserved |
| **MissingIndicator** | ➕ new column | ➕ new column | ➕ new column (int) | Adds a new binary feature |
| **SimpleImputer (mean / median / constant)** | ❌ unchanged | ❌ unchanged | ✅ real | Values stay on orig scale |
| **SimpleImputer (mode, categorical)** | ❌ unchanged | ❌ unchanged | ❌ unchanged (text) | Category set unchanged |
| **OneHotEncoder** | ➕ replaced by binary columns | ✅ encoded output is numeric | ✅ int / bool | Expands each category into 0/1 indicators |
| **OrdinalEncoder** | ✅ updated | ✅ encoded output is numeric | ✅ int / real | Ordered categories mapped to integer ranks |
| **Winsorizer / Clipper** | ✅ updated | ❌ unchanged | ✅ real | Values clipped to quantile bound |
| **OutlierDetector** | ❌ unchanged | ❌ unchanged | ❌ unchanded | Diagnostic only, no data modification |
| **log1p** | ✅ updated | ❌ unchanged | ✅ real | Nonlinear transform changes numeric range |
| **asinh** | ✅ updated | ❌ unchanged | ✅ real | Nonlinear transform changes numeric range |
| **Yeo–Johnson** | ✅ updated | ❌ unchanged | ✅ real | Power transform changes scale |
| **QuantileTransformerApprox** | ✅ updated | ❌ unchanged | ✅ real | Data mapped to fixed target distribution |
| **StandardScaler** | ✅ updated (≈ mean 0, std 1) | ❌ unchanged | ✅ real | Values rescaled |
| **RobustScalerApprox** | ✅ updated | ❌ unchanged | ✅ real | Values rescaled using median/IQR |
| **MinMaxScaler** | ✅ updated (fixed range) | ❌ unchanged | ✅ real | Explicit range mapping |

### Preprocessing Exceptions

The following algorithms must not use preprocessing because they are defined on the original data scale or on raw categorical counts. Applying preprocessing would change the mathematical definition and invalidate the results.

This includes:

CrossTabTable, ChiSquared, FisherExact, which operate on raw contingency tables.

Absolute / Relative difference, SMAPE, MAE, MSE, R², log_difference, which are metrics whose meaning changes if inputs are transformed.

StandardHistogram and MedianBasedOnHistogram, which are aggregation primitives and preprocessing building blocks, not consumers of preprocessing.

### 4. Basic Preprocessing Pipelines (per Variable)

Below are ready-to-use pipelines, built strictly following the allowed order.

#### A) Nominal Categorical Variables

[ SimpleImputer(mode | constant) → OneHotEncoder ]
[ MissingIndicator + SimpleImputer → OneHotEncoder ]

No scaling
No numeric transforms

#### B) Ordinal Variables

[ SimpleImputer(mode | median) → OrdinalEncoder ]
[ MissingIndicator + SimpleImputer → OrdinalEncoder ]

Optional (only if treated as numeric):
→ RobustScalerApprox

If treated as nominal (ignore ordering):
→ replace OrdinalEncoder with OneHotEncoder

#### C) Numerical – Discrete (counts)

Simple
MissingIndicator
→ SimpleImputer
→ StandardScaler

Skewed
MissingIndicator
→ SimpleImputer
→ log1p OR YeoJohnson
→ StandardScaler

Outlier-robust
MissingIndicator
→ SimpleImputer
→ Winsorizer
→ RobustScalerApprox

#### D) Numerical – Continuous

Standard
MissingIndicator
→ SimpleImputer
→ StandardScaler

Robust
MissingIndicator
→ SimpleImputer
→ RobustScalerApprox

Skewed / heavy-tailed
MissingIndicator
→ SimpleImputer
→ YeoJohnson OR asinh
→ StandardScaler

Distribution-shaping
MissingIndicator
→ SimpleImputer
→ QuantileTransformerApprox

**Key Rules**

- Preprocessing is per variable
- Different variables may have different pipelines
- A variable cannot have multiple power transforms
- Scaling never comes before imputation or transforms

______________________________________________________________________

## When to Apply Preprocessing (and When NOT to)

### Apply preprocessing when:

- You train **models that assume scale/shape properties**, e.g.:
  - linear/logistic/Cox regression
  - mixed-effects models (optimization stability)
  - clustering (KMeans)
- You need to handle missingness in predictors consistently across nodes.
- You want stable gradients / optimization and comparable coefficient magnitudes.

### Do NOT apply preprocessing when it changes the meaning of a statistical test:

Many “classical statistical functions” are defined on the **original scale** and transforming inputs
may change the interpretation.

Examples:

- **CrossTabTable / ChiSquared / FisherExact**: operate on categorical counts; scaling is irrelevant.
- **Absolute/Relative difference, SMAPE, log difference, MAE/MSE/R2**: these are evaluation metrics;
  transforming inputs changes the metric definition.
- **Correlation/Covariance**: optional (centering/scaling changes interpretation unless explicitly desired).

Rule of thumb:

- **Preprocessing is primarily for modeling / feature engineering**, not for descriptive tests,
  unless the test explicitly requires a transformation (rare).

### Scope of Preprocessing

**Rule A1 – Preprocessing is defined per variable**

- Each variable (column) has its own preprocessing pipeline.
- There is no single global preprocessing applied to all variables.
- Different variables within the same model MAY use different preprocessing pipelines.

**Rule A2 – Preprocessing is applied before modeling, not during aggregation**

- All preprocessing parameters (means, quantiles, lambdas, etc.) are computed federated.
- The resulting transformations are applied locally and deterministically.

## Summary

The federated preprocessing module in MIP provides a **reusable, federated-safe toolkit**
for privacy-preserving statistical modeling.

By separating preprocessing from modeling and by reusing standard MIP primitives
(`NumpyAggregator`, `StandardHistogram`), MIP enables:

- correctness
- stability
- reusability
- consistency across future statistical models
