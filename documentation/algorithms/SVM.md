# Linear SVM

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Classification method](#classification-method)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

## Overview

Linear support vector machines classify observations using a separating
hyperplane in the feature space. This implementation fits linear SVC models at
each site and averages the learned coefficients and intercepts.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `y` | Nominal target variable. |
| `x` | Numerical features. |

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `C` | Misclassification penalty and regularization strength. | `1.0` |
| `gamma` | Kernel coefficient passed to `sklearn.svm.SVC`. | `0.1` |

## Classification method

Each site fits:

```text
sklearn.svm.SVC(kernel="linear", C=C, gamma=gamma)
```

For binary classification, the decision function is:

```text
f(x) = w'x + b
```

For multiclass output from scikit-learn, class-specific coefficient rows and
intercepts are averaged within the site before cross-site averaging.

## Federated computation

The method does not optimize a single shared margin. Instead, it averages fitted
linear model parameters.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Target class union | Validate that at least two target classes are present. |
| Site-level coefficient vector | Contribute to the averaged weight vector. |
| Site-level intercept | Contribute to the averaged intercept. |
| Site observation count | Report total number of fitted observations. |

### Federated flow

```text
Input:
    y: class labels
    X: numerical features
    C: regularization parameter
    gamma: SVC gamma parameter

Step 1:
    Aggregate the union of target classes.

Step 2:
    Validate that at least two classes are present.

Step 3:
    At each site:
        fit sklearn SVC(kernel="linear")
        extract coefficient rows and intercepts
        average class-specific rows to one coefficient vector if needed

Step 4:
    Average site-level coefficient vectors and intercepts with equal site weight.

Output:
    averaged linear SVM parameters
```

## Technical decisions

- Only numerical features are supported.
- The method uses `sklearn.svm.SVC` with `kernel="linear"`.
- Site models are averaged equally, not weighted by observation count.
- This is parameter averaging, not a centralized linear SVM objective.
- Missing values are handled before fitting by the required missing-values
  preprocessing step.

## Outputs

| Field | Description |
|---|---|
| `title` | Result title. |
| `n_obs` | Total number of observations used across sites. |
| `weights` | Averaged linear coefficient vector. |
| `intercept` | Averaged intercept. |

## Validation against state-of-the-art implementation

The local fitting method is aligned with:

```text
sklearn.svm.SVC(kernel="linear")
```

The aggregated output is not equivalent to fitting one pooled scikit-learn SVM;
it averages fitted site-level parameters.

## Limitations and assumptions

- Features must be numerical.
- At least two target classes are required.
- Feature scaling is important for SVM behavior.
- Equal site weighting can differ from observation-weighted pooled fitting.
- The returned model parameters are averaged summaries and do not include
  support vectors or per-observation predictions.
