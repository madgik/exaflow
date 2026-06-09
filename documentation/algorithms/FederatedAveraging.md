# Federated Averaging

## Table of contents

- [Overview](#overview)
- [Inputs](#inputs)
  - [Required inputs](#required-inputs)
  - [Parameters](#parameters)
- [Method](#method)
- [Federated computation](#federated-computation)
  - [Aggregated quantities](#aggregated-quantities)
  - [Federated flow](#federated-flow)
- [Technical decisions](#technical-decisions)
- [Outputs](#outputs)
- [Validation against state-of-the-art implementation](#validation-against-state-of-the-art-implementation)
- [Limitations and assumptions](#limitations-and-assumptions)

## Overview

Federated averaging is a helper method for combining model parameters learned
separately at multiple sites. It computes an equal-weight average for each named
parameter array.

## Inputs

### Required inputs

| Input | Description |
|---|---|
| `params` | Dictionary of named numerical parameter arrays. |

### Parameters

No user parameters are exposed.

## Method

For each parameter `theta_j` contributed by `m` sites, the averaged parameter is:

```text
theta_bar = (theta_1 + theta_2 + ... + theta_m) / m
```

The same formula is applied independently to each named array after preserving
its original shape.

## Federated computation

Each site contributes one copy of each parameter array. Arrays are flattened for
aggregation, summed elementwise, reshaped, and divided by the number of
participating sites.

### Aggregated quantities

| Quantity | Purpose |
|---|---|
| Participation count | Number of sites contributing parameters. |
| Elementwise parameter sums | Numerator for averaged parameters. |

### Federated flow

```text
Input:
    params: dictionary of named arrays

Step 1:
    Each site contributes 1.0 to the participation count.

Step 2:
    For each named parameter:
        flatten the array
        aggregate elementwise sums
        reshape to the original array shape
        divide by the participation count

Output:
    dictionary of averaged parameter arrays
```

## Technical decisions

- All participating sites receive equal weight.
- Parameter arrays are averaged elementwise.
- Shapes are preserved in the returned dictionary.
- If no participants are counted, the input parameters are returned as lists.

## Outputs

| Field | Description |
|---|---|
| Parameter name | Averaged array for that parameter, returned as nested lists. |

## Validation against state-of-the-art implementation

The helper follows the standard FedAvg parameter-averaging formula. It is a
utility function rather than a standalone statistical estimator, so validation
is based on arithmetic equivalence of summed parameters divided by participant
count.

## Limitations and assumptions

- Parameters with the same name must have compatible shapes at all sites.
- Equal weighting may not match sample-size-weighted training objectives.
- Averaging independently trained parameters is not equivalent to fitting a
  single pooled model in general.
