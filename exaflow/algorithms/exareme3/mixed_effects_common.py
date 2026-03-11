from __future__ import annotations

from typing import Iterable

import numpy as np

from exaflow.worker_communication import BadUserInput


def split_grouping_var(
    x_vars: Iterable[str],
    grouping_var: str,
    metadata: dict,
) -> tuple[list[str], list[str]]:
    x_vars = list(x_vars)
    if grouping_var is None:
        raise BadUserInput("Parameter 'grouping_var' should not be blank.")
    if grouping_var not in x_vars:
        raise BadUserInput(
            "Parameter 'grouping_var' must match exactly one variable included in 'x'."
        )

    fixed_vars = [var for var in x_vars if var != grouping_var]
    if not fixed_vars:
        raise BadUserInput(
            "At least one fixed-effect covariate must remain in 'x' after removing 'grouping_var'."
        )

    categorical_vars = [var for var in fixed_vars if metadata[var]["is_categorical"]]
    numerical_vars = [var for var in fixed_vars if not metadata[var]["is_categorical"]]
    return categorical_vars, numerical_vars


def get_group_ids(data, grouping_var: str) -> np.ndarray:
    return data[grouping_var].astype(str).to_numpy(copy=False)


def encode_ordinal_response(series, category_order: list[str | int]) -> tuple[np.ndarray, list[str]]:
    if not isinstance(category_order, list) or len(category_order) < 2:
        raise BadUserInput(
            "Parameter 'category_order' must be a list with at least two ordered categories."
        )

    order = [str(value) for value in category_order]
    if len(order) != len(set(order)):
        raise BadUserInput("Parameter 'category_order' must not contain duplicates.")

    values = series.astype(str)
    observed = set(values.unique().tolist())
    allowed = set(order)
    if not observed.issubset(allowed):
        missing = sorted(observed - allowed)
        raise BadUserInput(
            f"Parameter 'category_order' does not cover all observed y categories: {missing}"
        )

    mapping = {label: idx for idx, label in enumerate(order)}
    encoded = values.map(mapping).to_numpy(dtype=int, copy=False)
    return encoded, order
