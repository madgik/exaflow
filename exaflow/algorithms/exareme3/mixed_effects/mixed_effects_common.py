from __future__ import annotations

from typing import Iterable

import numpy as np

from exaflow.worker_communication import BadUserInput


def normalize_grouping_vars(grouping_var: list[str]) -> list[str]:
    if not isinstance(grouping_var, list):
        raise BadUserInput("Parameter 'grouping_var' must be a list.")
    grouping_vars = grouping_var

    if not 1 <= len(grouping_vars) <= 2:
        raise BadUserInput(
            "Parameter 'grouping_var' must include one or two variables."
        )
    if any(not isinstance(var, str) or not var for var in grouping_vars):
        raise BadUserInput("Parameter 'grouping_var' must contain variable names.")
    if len(set(grouping_vars)) != len(grouping_vars):
        raise BadUserInput("Parameter 'grouping_var' must not contain duplicates.")
    return grouping_vars


def split_grouping_var(
    x_vars: Iterable[str],
    grouping_var: list[str],
    metadata: dict,
) -> tuple[list[str], list[str]]:
    x_vars = list(x_vars)
    grouping_vars = normalize_grouping_vars(grouping_var)
    missing = [var for var in grouping_vars if var not in x_vars]
    if missing:
        raise BadUserInput(
            "Parameter 'grouping_var' must match variables included in inputdata 'x'."
        )
    fixed_vars = [var for var in x_vars if var not in grouping_vars]
    if not fixed_vars:
        raise BadUserInput(
            "Inputdata 'Covariates and grouping variable' must include at least "
            "one fixed-effect covariate."
        )
    categorical_vars = [var for var in fixed_vars if metadata[var]["is_categorical"]]
    numerical_vars = [var for var in fixed_vars if not metadata[var]["is_categorical"]]
    return categorical_vars, numerical_vars


def get_group_ids(data, grouping_var: list[str]) -> np.ndarray:
    grouping_vars = normalize_grouping_vars(grouping_var)
    if len(grouping_vars) == 1:
        return data[grouping_vars[0]].astype(str).to_numpy(copy=False)

    group_frame = data[grouping_vars].astype(str)
    labels = group_frame.apply(
        lambda row: "".join(
            f"{len(var)}:{var}{len(row[var])}:{row[var]}" for var in grouping_vars
        ),
        axis=1,
    )
    return labels.to_numpy(copy=False)


def encode_ordinal_response(
    series, category_order: list[str | int]
) -> tuple[np.ndarray, list[str]]:
    if not isinstance(category_order, list) or len(category_order) < 2:
        raise BadUserInput(
            "Parameter 'category_order' must be a list with at least two "
            "ordered categories."
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
            "Parameter 'category_order' does not cover all observed y "
            f"categories: {missing}"
        )

    mapping = {label: idx for idx, label in enumerate(order)}
    encoded = values.map(mapping).to_numpy(dtype=int, copy=False)
    return encoded, order
