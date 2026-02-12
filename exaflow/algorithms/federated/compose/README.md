# Compose / Transformation Pipeline

## Column Transformer (FederatedColumnTransformer)

### Name

**Column Transformer (FederatedColumnTransformer)**

### Type

**Meta-Transformer** (preprocessing pipeline)

### Goal (Why we need it)

ColumnTransformer allows applying **different transformers to different column subsets** within a single pipeline, similar to sklearn's ColumnTransformer.
In a federated setting, we want the *same* transformation logic as centralized pipelines **without sharing raw data**, coordinating transformer fitting and application across column groups.

### When to use

Use ColumnTransformer when:

* you need to apply **different preprocessing** to different column groups
* you have both categorical and numerical features requiring different encodings
* you want a unified pipeline interface
* you want to compose multiple transformers (e.g., OneHot for categoricals, Passthrough for numericals)
* you need flexible column selection (by name, index, dtype, or callable selector)

### When NOT to use

Avoid / be careful when:

* all columns need the same transformation (use single transformer instead)
* transformation logic is very simple (may be overkill)

---

### Inputs / Outputs

| Item                      | Description                                                          |
| ------------------------- | -------------------------------------------------------------------- |
| **transformers**          | List of (name, transformer, columns) tuples                          |
| **data**                  | DataFrame to fit/transform                                           |
| **categorical_vars**      | List of categorical variable names                                   |
| **numerical_vars**        | Optional list of numerical variable names                            |
| **prefix_feature_names**  | Whether to prefix output features with transformer name (default: False) |
| **remainder**             | How to handle remaining columns: "drop" or "passthrough"             |
| **agg_client**            | Federated aggregation client                                         |

**Outputs**

* `transform(data)`: transformed design matrix as numpy array
* `get_feature_names_out()`: list of feature names in transformed output

### Key Differences from scikit-learn

| Aspect              | sklearn ColumnTransformer     | MIP Federated Implementation  |
| ------------------- | ----------------------------- | ----------------------------- |
| Data access         | Full centralized data         | Data remains local per client |
| Column selection    | Same (name/index/dtype/func)  | Same                          |
| Transformer types   | Any sklearn transformer       | Only federated transformers   |
| Sparse output       | Supported                     | Dense only                    |
| Remainder handling  | drop/passthrough/transformer  | drop/passthrough (num only)   |
| Performance         | CPU/memory bound              | Network + aggregation bound   |

### Approximation vs Exactness

| Component               | sklearn | MIP                    |
| ----------------------- | ------- | ---------------------- |
| Column selection logic  | Exact   | Exact                  |
| Transformer application | Exact   | Exact (per transformer)|
| Feature concatenation   | Exact   | Exact                  |

---

## make_column_selector

### Name

**make_column_selector**

### Type

**Utility Function** (column selection helper)

### Goal (Why we need it)

Creates a callable column selector that filters columns by **dtype** (similar to sklearn's make_column_selector).

### When to use

Use make_column_selector when:

* you want to select columns by dtype in ColumnTransformer
* you need to differentiate numeric vs categorical features automatically
* you want flexible dtype-based selection (include/exclude patterns)

### When NOT to use

* When column names are known statically (use explicit column lists)

---

### Inputs / Outputs

| Item              | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| **dtype_include** | Optional list of dtypes to include (e.g., ["number"])      |
| **dtype_exclude** | Optional list of dtypes to exclude (e.g., ["object"])      |

**Outputs**

* Returns a callable that takes a DataFrame and returns list of column names matching criteria

### Key Differences from scikit-learn

| Aspect         | sklearn make_column_selector | MIP  |
| -------------- | ---------------------------- | ---- |
| Functionality  | Same                         | Same |
| Dtype matching | Same                         | Same |

### Approximation vs Exactness

| Component      | sklearn | MIP   |
| -------------- | ------- | ----- |
| Dtype matching | Exact   | Exact |
