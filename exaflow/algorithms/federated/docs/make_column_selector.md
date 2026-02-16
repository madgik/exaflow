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

