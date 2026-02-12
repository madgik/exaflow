# Naive Bayes Algorithms

## Table of Contents

1. [Gaussian Naive Bayes](#gaussian-naive-bayes-federatedgaussiannb) - For continuous features
2. [Categorical Naive Bayes](#categorical-naive-bayes-federatedcategoricalnb) - For discrete features

---

## Gaussian Naive Bayes (FederatedGaussianNB)

### Name

**Gaussian Naive Bayes (FederatedGaussianNB)**

### Type

**Probabilistic Classifier** (supervised classification)

### Goal (Why we need it)

Gaussian Naive Bayes assumes features are **conditionally independent given the class** and follow **Gaussian distributions**. It learns class-conditional means and variances to predict class probabilities.
In a federated setting, we want the *same* class priors, means, and variances as centralized Gaussian NB **without sharing raw data**, using only aggregated sufficient statistics (sums, sums of squares, counts per class).

### When to use

Use Gaussian Naive Bayes when:

* you need **multi-class classification** with probabilistic outputs
* features are **continuous/numerical** and approximately Gaussian per class
* you have small to moderate sample sizes
* features are (approximately) conditionally independent given the class
* you want a fast, simple baseline classifier

### When NOT to use

Avoid / be careful when:

* features are categorical (use Categorical Naive Bayes instead)
* features are highly correlated (violates independence assumption)
* feature distributions are very non-Gaussian (consider transformations)
* you need complex decision boundaries (NB assumes simple boundaries)
* privacy is critical beyond aggregate statistics

---

### Inputs / Outputs

| Item              | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| **X**             | Feature matrix, shape `(n_local, p)`                         |
| **y**             | Class labels, shape `(n_local,)`                             |
| **x_vars**        | List of feature variable names                               |
| **labels**        | List of expected class labels                                |
| **var_smoothing** | Variance smoothing parameter (default: 1e-9)                 |
| **agg_client**    | Federated aggregation client                                 |

**Outputs (FederatedGaussianNBResults)**

* `theta`: class-conditional means, shape `(n_classes, n_features)`
* `var`: class-conditional variances, shape `(n_classes, n_features)`
* `class_count`: number of samples per class
* `class_prior`: prior probabilities for each class
* `labels`: list of class labels
* `predict_proba(X)`: predicted class probabilities
* `predict(X)`: predicted class labels

### Key Differences from scikit-learn

| Aspect                | scikit-learn          | MIP Federated Implementation  |
| --------------------- | --------------------- | ----------------------------- |
| Data access           | Full centralized data | Data remains local per client |
| Classes discovery     | Auto-discovered       | Pre-specified in constructor  |
| Sufficient statistics | Local computation     | Federated sums/sums_sq/counts |
| Variance smoothing    | Same formula          | Same formula                  |
| Performance           | CPU/memory bound      | Network + aggregation bound   |

### Approximation vs Exactness

| Component             | scikit-learn | MIP   |
| --------------------- | ------------ | ----- |
| Class priors          | Exact        | Exact |
| Class-conditional μ   | Exact        | Exact |
| Class-conditional σ²  | Exact        | Exact |
| Variance smoothing    | Exact        | Exact |
| Probability estimates | Exact        | Exact |

---

## Categorical Naive Bayes (FederatedCategoricalNB)

### Name

**Categorical Naive Bayes (FederatedCategoricalNB)**

### Type

**Probabilistic Classifier** (supervised classification)

### Goal (Why we need it)

Categorical Naive Bayes is designed for **discrete/categorical features**. It learns the frequency of each category-value combination within each class using **smoothed categorical distributions**.
In a federated setting, we want the *same* class priors and category probabilities as centralized Categorical NB **without sharing raw data**, using only aggregated category counts per class.

### When to use

Use Categorical Naive Bayes when:

* you need **multi-class classification** with categorical features
* features are **discrete/categorical** (not continuous)
* categorical features are ordinal-encoded or integer-coded
* you have small to moderate sample sizes
* features are (approximately) conditionally independent given the class

### When NOT to use

Avoid / be careful when:

* features are continuous (use Gaussian Naive Bayes instead)
* features are highly correlated (violates independence assumption)
* categories are not properly encoded as integers
* you have many rare categories (sparse counts)
* privacy is critical beyond aggregate statistics

---

### Inputs / Outputs

| Item             | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| **X**            | Encoded feature matrix (integer codes), shape `(n_local, p)`     |
| **y**            | Class labels, shape `(n_local,)`                                 |
| **y_var**        | Name of the target variable                                      |
| **x_vars**       | List of categorical feature variable names                       |
| **categories**   | Dict mapping variable names to ordered category lists            |
| **alpha**        | Laplace smoothing parameter (default: 1.0)                       |
| **agg_client**   | Federated aggregation client                                     |

**Outputs (FederatedCategoricalNBResults)**

* `class_count`: number of samples per class
* `category_count`: dict of category counts per class per feature
* `class_log_prior`: log prior probabilities for each class
* `category_log_prob`: dict of log probabilities per category per class
* `labels`: list of class labels
* `predict(X)`: predicted class labels

### Key Differences from scikit-learn

| Aspect                | scikit-learn CategoricalNB | MIP Federated Implementation  |
| --------------------- | -------------------------- | ----------------------------- |
| Data access           | Full centralized data      | Data remains local per client |
| Categories discovery  | Auto-discovered            | Pre-specified via metadata    |
| Sufficient statistics | Local computation          | Federated category counts     |
| Smoothing             | Laplace (alpha)            | Laplace (alpha)               |
| Unknown categories    | Ignored (treated as 0)     | Ignored (treated as 1)        |
| Performance           | CPU/memory bound           | Network + aggregation bound   |

### Approximation vs Exactness

| Component                 | scikit-learn | MIP   |
| ------------------------- | ------------ | ----- |
| Class priors              | Exact        | Exact |
| Category frequencies      | Exact        | Exact |
| Laplace smoothing         | Exact        | Exact |
| Log-probability estimates | Exact        | Exact |
| Unknown category handling | Exact        | Exact |
