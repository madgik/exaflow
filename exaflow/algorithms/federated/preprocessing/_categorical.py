"""Categorical imputation helpers using trusted, predeclared metadata only."""

from math import isfinite

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils import BadInputError


def category_key(value):
    """Keep numeric codes, text codes and booleans distinct."""
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (str, int, float, bool):
        raise BadInputError("Categories must be finite JSON scalar values.")
    if isinstance(value, float) and not isfinite(value):
        raise BadInputError("Categories must be finite JSON scalar values.")
    return ("number" if type(value) in (int, float) else type(value).__name__), value


def validate_vocabulary(categories):
    if not isinstance(categories, (list, tuple)) or not categories:
        raise BadInputError("Categorical most_frequent requires a metadata vocabulary.")
    validated = []
    for column in categories:
        if not isinstance(column, (list, tuple)) or not column:
            raise BadInputError("Provide a non-empty metadata vocabulary per feature.")
        keys = [category_key(value) for value in column]
        if len(set(keys)) != len(keys):
            raise BadInputError("Metadata vocabulary entries must be unique.")
        validated.append([key[1] for key in keys])
    return validated


def vocabulary_counts(values, categories):
    """Validate every observed value before any frequency counts are shared."""
    if values.ndim != 2 or values.shape[1] != len(categories):
        raise BadInputError("Metadata vocabulary must match the feature count.")
    counts = []
    for idx, column in enumerate(categories):
        positions = {category_key(value): i for i, value in enumerate(column)}
        local = np.zeros(len(column), dtype=float)
        for value in values[:, idx]:
            if value is None or value is pd.NA:
                continue
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                continue
            key = category_key(value)
            if key not in positions:
                raise BadInputError(
                    "Observed values are outside the metadata vocabulary."
                )
            local[positions[key]] += 1
        counts.append(local)
    return counts
