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

