"""Shared numerical utilities for fitted numerical transformers."""

from typing import Any

import numpy as np

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import as_numeric_array
from exaflow.algorithms.federated.utils.agg_client import AggregationClient


def reject_infinite(array: np.ndarray) -> None:
    if np.isinf(array).any():
        raise BadInputError("Input must not contain infinite values.")


def as_2d_numeric(data: Any, *, name: str = "Input") -> tuple[np.ndarray, bool]:
    array = as_numeric_array(data, input_name=name)
    reject_infinite(array)
    if array.ndim == 1:
        return array.reshape(-1, 1).copy(), True
    if array.ndim != 2:
        raise BadInputError(f"{name} must be a one- or two-dimensional array.")
    if array.shape[1] == 0:
        raise BadInputError(f"{name} must contain at least one feature.")
    return array.copy(), False


def require_rows(array: np.ndarray) -> None:
    if array.shape[0] == 0:
        raise BadInputError("Input must contain at least one observation.")


def global_sum_count(
    array: np.ndarray, agg_client: AggregationClient
) -> tuple[np.ndarray, np.ndarray]:
    valid = ~np.isnan(array)
    local_sum = np.nansum(array, axis=0).astype(float)
    local_count = valid.sum(axis=0).astype(float)
    total_sum = np.asarray(agg_client.sum(local_sum), dtype=float)
    total_count = np.asarray(agg_client.sum(local_count), dtype=float)
    return total_sum, total_count


def global_population_scale(
    array: np.ndarray, agg_client: AggregationClient
) -> tuple[np.ndarray, np.ndarray]:
    total_sum, total_count = global_sum_count(array, agg_client)
    if np.any(total_count <= 0):
        raise BadInputError("At least one feature has no valid observations.")
    mean = total_sum / total_count
    deviations = array - mean
    local_statistics = np.concatenate(
        (np.nansum(deviations**2, axis=0), np.nansum(deviations, axis=0))
    )
    total_squared, correction = np.split(
        np.asarray(agg_client.sum(local_statistics), dtype=float), 2
    )
    variance = np.maximum(
        (total_squared - correction**2 / total_count) / total_count, 0.0
    )
    eps = np.finfo(float).eps
    error_bound = total_count * eps * variance + (total_count * mean * eps) ** 2
    scale = np.where(variance <= error_bound, 1.0, np.sqrt(variance))
    return mean, scale
