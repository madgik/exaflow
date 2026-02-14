## 2. descriptive_stats.py - FederatedDescriptiveStatistics

### Name

**Descriptive Statistics Wrapper (FederatedDescriptiveStatistics)**

### Type

**Convenience Wrapper** (unified interface)

### Goal (Why we need it)

Provides a **unified interface** that combines describe, histogram and correlation functions into a statsmodels-style API.
It is a wrapper class that calls the underlying algorithms (FederatedDescribe, FederatedHistogram, FederatedPearsonCorrelation).

### When to use

Use FederatedDescriptiveStatistics when:

* you want a **unified API** for multiple descriptive statistics
* you prefer a statsmodels-style interface
* you need quick access to describe, hist, corrcoef through one object

### When NOT to use

* When you only need one of the underlying functions (use it directly)

---

### Inputs / Outputs

**Available Methods:**

* `describe()`: Calls FederatedDescribe
* `hist()`: Calls FederatedHistogram
* `corrcoef()`: Calls FederatedPearsonCorrelation
* `pearson_correlation()`: Alias for corrcoef()

Inputs/Outputs are the same as the underlying algorithms.

### Key Differences from statsmodels

| Aspect      | statsmodels           | MIP Implementation         |
| ----------- | --------------------- | -------------------------- |
| Purpose     | Full statistical API  | Convenience wrapper only   |
| Scope       | Broader functionality | Descriptive stats focused  |

### Approximation vs Exactness

Inherits exactness/approximation from the underlying algorithms.

---


