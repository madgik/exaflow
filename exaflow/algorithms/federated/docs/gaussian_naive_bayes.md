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
| Class-conditional ΞΌ   | Exact        | Exact |
| Class-conditional ΟƒΒ²  | Exact        | Exact |
| Variance smoothing    | Exact        | Exact |
| Probability estimates | Exact        | Exact |

---


