# Preprocessing Algorithms

## Table of Contents

1. [One-Hot Encoder](#one-hot-encoder-federatedonehotencoder) - Encode nominal categorical features
2. [Ordinal Encoder](#ordinal-encoder-federatedordinalencoder) - Encode ordinal categorical features
3. [Passthrough](#passthrough-federatedpassthrough) - Identity transformer

---

## One-Hot Encoder (FederatedOneHotEncoder)

### Name

**One-Hot Encoder (FederatedOneHotEncoder)**

### Type

**Preprocessing** (feature encoding)

### Goal (Why we need it)

One-Hot Encoding transforms **categorical features** into **binary dummy variables** (one column per category, minus the first category as reference).
In a federated setting, we want the *same* encoding scheme as centralized one-hot encoding **without sharing raw data**, using only aggregated category discovery via union operation.

### When to use

Use One-Hot Encoder when:

* you have **categorical features** that need numeric encoding
* you want to use categorical features in linear models (OLS, Logistic Regression)
* categories are unordered (nominal data)
* you can accept the increase in dimensionality
* you want to avoid imposing ordinal relationships

### When NOT to use

Avoid / be careful when:

* categories are ordered/ordinal (use Ordinal Encoder instead)
* very high cardinality features (creates too many columns)
* categories are rare (creates sparse columns with little information)
* you need the intercept term (this implementation drops first category)
* tree-based models are used (they can handle categorical features directly)

---

### Inputs / Outputs

| Item                 | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| **data**             | DataFrame with categorical and numerical features          |
| **categorical_vars** | List of categorical variable names to encode               |
| **numerical_vars**   | Optional list of numerical variable names to pass through  |
| **agg_client**       | Federated aggregation client                               |

**Outputs**

* `dummy_categories`: dict mapping each categorical variable to its dummy categories (all except first)
* `transform(data)`: encoded design matrix as numpy array
* `get_feature_names_out()`: list of feature names in encoded output

### Key Differences from scikit-learn

| Aspect               | scikit-learn OneHotEncoder    | MIP Federated Implementation  |
| -------------------- | ----------------------------- | ----------------------------- |
| Data access          | Full centralized data         | Data remains local per client |
| Category discovery   | Local observation             | Federated union aggregation   |
| Drop strategy        | Configurable (first/if_binary)| Always drops first            |
| Sparse output        | Supported                     | Dense only                    |
| Unknown categories   | Error or ignore               | Implicitly ignored (zero)     |
| Performance          | CPU/memory bound              | Network + aggregation bound   |

### Approximation vs Exactness

| Component           | scikit-learn | MIP              |
| ------------------- | ------------ | ---------------- |
| Category discovery  | Exact        | Exact (via union)|
| Encoding            | Exact        | Exact            |
| Dummy creation      | Exact        | Exact            |
| Drop first category | Exact        | Exact            |

---

## Ordinal Encoder (FederatedOrdinalEncoder)

### Name

**Ordinal Encoder (FederatedOrdinalEncoder)**

### Type

**Preprocessing** (feature encoding)

### Goal (Why we need it)

Ordinal Encoding maps **ordered categorical features** to **integer codes** (0, 1, 2, ...) based on a predefined ordering.
In a federated setting, categories and their order are **pre-specified via metadata** (not learned from data), ensuring consistent encoding across all sites without data sharing.

### When to use

Use Ordinal Encoder when:

* you have **ordinal/ordered categorical features** (e.g., low/medium/high)
* the ordering is known and meaningful
* you want compact integer representations
* features will be used in models that respect ordinal relationships
* category lists are available in metadata

### When NOT to use

Avoid / be careful when:

* categories are unordered/nominal (use One-Hot Encoder instead)
* ordering is unknown or arbitrary
* categories are not pre-specified in metadata
* you need to discover categories from data (this implementation requires pre-specification)

---

### Inputs / Outputs

| Item                 | Description                                                      |
| -------------------- | ---------------------------------------------------------------- |
| **data**             | DataFrame with categorical features                              |
| **categorical_vars** | List of categorical variable names to encode                     |
| **numerical_vars**   | Optional list of numerical variable names to pass through        |
| **categories**       | Dict mapping variable names to ordered category lists (required) |
| **handle_unknown**   | How to handle unknown categories: "ignore" or "error"            |
| **unknown_value**    | Integer code for unknown categories (default: -1)                |
| **agg_client**       | Federated aggregation client (not used, included for interface)  |

**Outputs**

* `categories_`: dict mapping each categorical variable to its ordered categories
* `transform(data)`: encoded matrix with integer codes as numpy array
* `get_feature_names_out()`: list of feature names in encoded output

### Key Differences from scikit-learn

| Aspect              | scikit-learn OrdinalEncoder   | MIP Federated Implementation  |
| ------------------- | ----------------------------- | ----------------------------- |
| Data access         | Full centralized data         | Data remains local per client |
| Category discovery  | Can learn from data           | Must pre-specify via metadata |
| Unknown handling    | Error or specified value      | Error or specified value      |
| Ordering            | Learned or specified          | Always specified              |
| Performance         | CPU/memory bound              | Local only (no aggregation)   |

### Approximation vs Exactness

| Component        | scikit-learn | MIP                      |
| ---------------- | ------------ | ------------------------ |
| Encoding         | Exact        | Exact                    |
| Category order   | Exact        | Exact (pre-specified)    |
| Unknown handling | Exact        | Exact                    |

---

## Passthrough (FederatedPassthrough)

### Name

**Passthrough (FederatedPassthrough)**

### Type

**Preprocessing** (identity transform)

### Goal (Why we need it)

Passthrough is a **no-op transformer** that simply returns data unchanged. It is useful in pipelines and column transformers when certain columns should not be transformed.

### When to use

Use Passthrough when:

* you want to include certain features without transformation in a pipeline
* using ColumnTransformer with some columns requiring no processing
* maintaining API consistency across transformers

### When NOT to use

* When actual transformation is needed (use appropriate transformer instead)

---

### Inputs / Outputs

| Item                 | Description                                       |
| -------------------- | ------------------------------------------------- |
| **data**             | DataFrame with features                           |
| **categorical_vars** | List of categorical variable names (ignored)      |
| **numerical_vars**   | List of numerical variable names to pass through  |

**Outputs**

* `transform(data)`: original numerical data as numpy array
* `get_feature_names_out()`: list of numerical variable names

### Key Differences from scikit-learn

| Aspect      | scikit-learn FunctionTransformer | MIP Federated Implementation |
| ----------- | -------------------------------- | ---------------------------- |
| Purpose     | Identity transform               | Identity transform           |
| API         | General callable                 | Specific to fed interface    |

### Approximation vs Exactness

| Component | scikit-learn | MIP   |
| --------- | ------------ | ----- |
| Transform | Exact (none) | Exact |
