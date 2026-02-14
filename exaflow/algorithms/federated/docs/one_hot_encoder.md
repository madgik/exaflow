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


