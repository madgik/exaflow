"""Shared numerical reductions used by feature and outlier transformers."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.statistics.percentile import Percentile
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import as_numeric_array
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


def _as_2d_numeric(data: Any, *, name: str = "Input") -> tuple[np.ndarray, bool]:
    array = as_numeric_array(data, input_name=name)
    _reject_infinite(array)
    if array.ndim == 1:
        return array.reshape(-1, 1).copy(), True
    if array.ndim != 2:
        raise BadInputError(f"{name} must be a one- or two-dimensional array.")
    if array.shape[1] == 0:
        raise BadInputError(f"{name} must contain at least one feature.")
    return array.copy(), False


def _restore_shape(array: np.ndarray, squeezed: bool) -> np.ndarray:
    return array[:, 0] if squeezed else array


def _require_rows(array: np.ndarray) -> None:
    if array.shape[0] == 0:
        raise BadInputError("Input must contain at least one observation.")


def _reject_infinite(array: np.ndarray) -> None:
    if np.isinf(array).any():
        raise BadInputError("Input must not contain infinite values.")


def _validate_local_domain(array: np.ndarray, *, predicate, message: str) -> None:
    invalid = np.isinf(array) | (~np.isnan(array) & ~predicate(array))
    if invalid.any():
        raise BadInputError(message)


def _global_sum_count(
    array: np.ndarray, agg_client: AggregationClient
) -> tuple[np.ndarray, np.ndarray]:
    valid = ~np.isnan(array)
    local_sum = np.nansum(array, axis=0).astype(float)
    local_count = valid.sum(axis=0).astype(float)
    total_sum = np.asarray(agg_client.sum(local_sum), dtype=float)
    total_count = np.asarray(agg_client.sum(local_count), dtype=float)
    return total_sum, total_count


def _global_mean(array: np.ndarray, agg_client: AggregationClient) -> np.ndarray:
    total_sum, total_count = _global_sum_count(array, agg_client)
    if np.any(total_count <= 0):
        raise BadInputError("At least one feature has no valid observations.")
    return total_sum / total_count


def _global_min_max(
    array: np.ndarray, agg_client: AggregationClient
) -> tuple[np.ndarray, np.ndarray]:
    aggregator = NumpyAggregator(agg_client)
    minimum = np.asarray(aggregator.global_min(array), dtype=float)
    maximum = np.asarray(aggregator.global_max(array), dtype=float)
    if np.any(~np.isfinite(minimum)) or np.any(~np.isfinite(maximum)):
        raise BadInputError("At least one feature has no valid observations.")
    return minimum, maximum


def _global_population_scale(
    array: np.ndarray, agg_client: AggregationClient
) -> tuple[np.ndarray, np.ndarray]:
    total_sum, total_count = _global_sum_count(array, agg_client)
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


def _validate_domain(
    array: np.ndarray, agg_client: AggregationClient, *, predicate, message: str
) -> None:
    invalid = np.isinf(array) | (~np.isnan(array) & ~predicate(array))
    count = np.asarray(agg_client.sum(np.array([invalid.sum()], dtype=float)))
    if count[0] > 0:
        raise BadInputError(message)


def _percentile(array: np.ndarray, q: float, agg_client: AggregationClient) -> float:
    result = Percentile(NumpyAggregator(agg_client)).compute(pd.Series(array), q)
    if result is None or not np.isfinite(result[0]):
        raise BadInputError("Percentile cannot be computed from the input.")
    return float(result[0])


def _global_quartiles(array, agg_client):
    return np.asarray(
        [
            [_percentile(array[:, i], q, agg_client) for i in range(array.shape[1])]
            for q in (0.25, 0.5, 0.75)
        ]
    )


def check_common_configuration(configuration, agg_client):
    """Compare metadata/settings fingerprints, never observed data values.

    All participants enter the same fixed-size check. SHA-256 words are exactly
    representable in float64; observed categories must never enter this hash.
    """
    encoded = json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
    fingerprint = np.frombuffer(sha256(encoded).digest(), dtype=">u4").astype(float)
    minimum = np.asarray(agg_client.min(fingerprint))
    maximum = np.asarray(agg_client.max(fingerprint))
    if not np.array_equal(minimum, maximum):
        raise BadInputError(
            "Participants must use the same ordered metadata and settings."
        )
