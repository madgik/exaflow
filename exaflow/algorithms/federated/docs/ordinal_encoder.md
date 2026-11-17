## Ordinal Encoder (\_FederatedOrdinalEncoder)

> **Storage and coding.** Categorical CDEs are stored as integer codes (binary `0`/`1`, nominal and ordinal from `1`; `dataset` and `visitid` included; subject identifiers are discrete integers) and reach this code as text codes. See [Data Types, Storage and Variable Coding](../../../../documentation/data-types-and-coding.md).

Implementation: [feature_engineering/\_ordinal_encoder.py](../feature_engineering/_ordinal_encoder.py).

> **Internal.** `_FederatedOrdinalEncoder` is a private building block hardcoded inside the algorithms that need it. It is not exported, not an algorithm or preprocessing step, and not exposed by any specification or HTTP endpoint. Binary and ordinal variables are never produced from nominal ones with it.

> **Covariate coding.** The regression, Cox and mixed-model algorithms code `ordinal` covariates with this encoder (integers `0..K-1` in the declared level order) and `binary` covariates with it as one 0/1 column (first declared level 0). `exareme3/utils/covariate_encoding.py` (`build_covariate_transformer`) is the single place that picks the coding by CDE type. With `handle_unknown="error"` a missing value or an undeclared level raises `BadInputError`.

### Name

**Ordinal Encoder (\_FederatedOrdinalEncoder)**

### Type

**Preprocessing** (feature encoding)

### Goal (Why we need it)

Ordinal Encoding maps **ordered categorical features** to **integer codes** (0, 1, 2, ...) based on a predefined ordering.
In a federated setting, categories and their order are **pre-specified via metadata** (not learned from data), ensuring consistent encoding across all sites without data sharing.

### When to use

Use Ordinal Encoder when:

- you have **ordinal/ordered categorical features** (e.g., low/medium/high)
- the ordering is known and meaningful
- you want compact integer representations
- features will be used in models that respect ordinal relationships
- category lists are available in metadata

### When NOT to use

Avoid / be careful when:

- categories are unordered/nominal (use One-Hot Encoder instead)
- ordering is unknown or arbitrary
- categories are not pre-specified in metadata
- you need to discover categories from data (this implementation requires pre-specification)

______________________________________________________________________

### Inputs / Outputs

| Item | Description |
| -------------------- | ---------------------------------------------------------------- |
| **data** | DataFrame with categorical features |
| **categorical_vars** | List of categorical variable names to encode |
| **numerical_vars** | Optional list of numerical variable names to pass through |
| **categories** | Dict mapping variable names to ordered category lists (required) |
| **handle_unknown** | How to handle unknown categories: "ignore" or "error" |
| **unknown_value** | Integer code for unknown categories (default: -1) |
| **agg_client** | Federated aggregation client (not used, included for interface) |

**Outputs**

- `categories_`: dict mapping each categorical variable to its ordered categories
- `transform(data)`: encoded matrix with integer codes as numpy array
- `get_feature_names_out()`: list of feature names in encoded output

### Key Differences from scikit-learn

| Aspect | scikit-learn OrdinalEncoder | MIP Federated Implementation |
| ------------------- | ----------------------------- | ----------------------------- |
| Data access | Full centralized data | Data remains local per client |
| Category discovery | Can learn from data | Must pre-specify via metadata |
| Unknown handling | Error or specified value | Error or specified value |
| Ordering | Learned or specified | Always specified |
| Performance | CPU/memory bound | Local only (no aggregation) |

### Approximation vs Exactness

| Component | scikit-learn | MIP |
| ---------------- | ------------ | ------------------------ |
| Encoding | Exact | Exact |
| Category order | Exact | Exact (pre-specified) |
| Unknown handling | Exact | Exact |

______________________________________________________________________
