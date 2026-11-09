"""Federated mathematical preprocessing primitives.

The classes in this module operate on local NumPy partitions.  Fitted
parameters are obtained through the existing AggregationClient operations.
Categorical imputation uses a trusted vocabulary and aggregates counts only.
"""

from __future__ import annotations

from numbers import Integral
from numbers import Number
from typing import Any
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import ndtri

from exaflow.algorithms.federated.preprocessing._batched_quantiles import (
    batched_quantiles,
)
from exaflow.algorithms.federated.preprocessing._categorical import validate_vocabulary
from exaflow.algorithms.federated.preprocessing._categorical import vocabulary_counts
from exaflow.algorithms.federated.preprocessing._configuration import (
    check_common_configuration,
)
from exaflow.algorithms.federated.preprocessing._yeo_johnson_search import (
    adaptive_lambda_search,
)
from exaflow.algorithms.federated.preprocessing._yeo_johnson_search import (
    safe_lambda_bounds,
)
from exaflow.algorithms.federated.preprocessing.recipes import OutlierDetectorRecipe
from exaflow.algorithms.federated.preprocessing.recipes import QuantileTransformerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import ScalerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import SimpleImputerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import WinsorizerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import YeoJohnsonRecipe
from exaflow.algorithms.federated.statistics.percentile import Percentile
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import as_numeric_array
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from exaflow.algorithms.federated.validation import check_choice
from exaflow.algorithms.federated.validation import check_fitted
from exaflow.algorithms.federated.validation import check_interval
from exaflow.algorithms.federated.validation import check_scalar


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


class _BaseTransformer:
    recipe_: Any = None

    def fit_transform(self, data, *, agg_client):
        return self.fit(data, agg_client=agg_client).transform(data)


class SimpleImputer(_BaseTransformer):
    """Impute missing values using globally fitted statistics.

    Constant imputation defaults to ``0`` for numeric input when no fill value
    is supplied. Categorical mode requires one ordered vocabulary per feature,
    supplied by a trusted metadata-aware caller. Unknown values are rejected;
    the primitive never discovers categories from observed data.
    """

    def __init__(
        self, strategy: str = "mean", fill_value: Any = None, *, categories=None
    ):
        check_choice(
            strategy, ("mean", "median", "constant", "most_frequent"), name="strategy"
        )
        self.strategy = strategy
        self.fill_value = fill_value
        self.categories = categories
        self.statistics_: Optional[np.ndarray] = None

    @classmethod
    def from_recipe(cls, recipe):
        recipe = SimpleImputerRecipe.model_validate(recipe)
        imputer = cls(strategy=recipe.strategy, categories=recipe.categories)
        imputer.statistics_ = np.asarray(recipe.statistics, dtype=object)
        imputer._numeric_output = recipe.numeric_output
        imputer.categories_ = recipe.categories
        imputer.recipe_ = recipe
        return imputer

    def fit(self, data, *, agg_client):
        if self.strategy == "constant":
            values = np.asarray(data, dtype=object)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.ndim != 2:
                raise BadInputError("Input must be one- or two-dimensional.")
            _require_rows(values)
            if values.shape[1] == 0:
                raise BadInputError("Input must contain at least one feature.")
            observed = values[~pd.isna(values)]
            self._numeric_output = all(isinstance(value, Number) for value in observed)
            fill_value = self.fill_value
            if fill_value is None:
                if not self._numeric_output:
                    raise BadInputError(
                        "fill_value is required for non-numeric constant imputation."
                    )
                fill_value = 0
            if self._numeric_output:
                numeric, _ = _as_2d_numeric(data)
                _reject_infinite(numeric)
                check_scalar(fill_value, name="fill_value")
            self.statistics_ = np.asarray([fill_value] * values.shape[1], dtype=object)
        elif self.strategy in {"mean", "median"}:
            self._numeric_output = True
            array, _ = _as_2d_numeric(data)
            _require_rows(array)
            _reject_infinite(array)
            if self.strategy == "mean":
                self.statistics_ = _global_mean(array, agg_client)
            else:
                self.statistics_ = np.asarray(
                    [
                        _percentile(array[:, idx], 0.5, agg_client)
                        for idx in range(array.shape[1])
                    ]
                )
        else:
            self._numeric_output = False
            # Reject schema/value errors collectively before anyone sends counts.
            errors = np.zeros(2, dtype=float)
            try:
                categories = validate_vocabulary(self.categories)
                values = np.asarray(data, dtype=object)
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                if values.ndim != 2 or values.shape[1] != len(categories):
                    raise BadInputError("Vocabulary must match the feature count.")
            except (BadInputError, TypeError, ValueError):
                errors[0] = 1
            if not errors[0]:
                try:
                    local_counts = vocabulary_counts(values, categories)
                except BadInputError:
                    errors[1] = 1
            global_errors = np.asarray(agg_client.sum(errors), dtype=float)
            if global_errors[0]:
                raise BadInputError(
                    "Categorical most_frequent requires a complete metadata vocabulary."
                )
            if global_errors[1]:
                raise BadInputError(
                    "Observed values are outside the metadata vocabulary."
                )
            check_common_configuration(categories, agg_client)
            statistics = []
            for column, local in zip(categories, local_counts):
                counts = np.asarray(agg_client.sum(local), dtype=float)
                if counts.sum() <= 0:
                    raise BadInputError(
                        "At least one feature has no valid observations."
                    )
                statistics.append(column[int(np.argmax(counts))])
            self.categories_ = categories
            self.statistics_ = np.asarray(statistics, dtype=object)
        self.recipe_ = SimpleImputerRecipe(
            strategy=self.strategy,
            statistics=self.statistics_.tolist(),
            numeric_output=self._numeric_output,
            categories=self.categories_ if self.strategy == "most_frequent" else None,
        )
        return self

    def transform(self, data):
        check_fitted(self, "statistics_")
        if self.statistics_ is None:
            raise BadInputError("SimpleImputer is not fitted.")
        values = np.asarray(data, dtype=object).copy()
        squeezed = values.ndim == 1
        if squeezed:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != len(self.statistics_):
            raise BadInputError(
                "Input feature count does not match the fitted imputer."
            )
        if self.strategy == "most_frequent":
            vocabulary_counts(values, self.categories_)
        for idx, statistic in enumerate(self.statistics_):
            missing = pd.isna(values[:, idx])
            values[missing, idx] = statistic
        if self._numeric_output:
            values, _ = _as_2d_numeric(values)
            _reject_infinite(values)
        return values[:, 0] if squeezed else values


class MissingIndicator(_BaseTransformer):
    def fit(self, data, *, agg_client=None):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        self._n_features = array.shape[1]
        return self

    def transform(self, data):
        check_fitted(self, "_n_features")
        array, squeezed = _as_2d_numeric(data)
        if getattr(self, "_n_features", None) != array.shape[1]:
            raise BadInputError(
                "Input feature count does not match the fitted indicator."
            )
        result = np.isnan(array).astype(int)
        return _restore_shape(result, squeezed)


class Log1pTransformer(_BaseTransformer):
    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _validate_domain(
            array,
            agg_client,
            predicate=lambda x: x >= 0,
            message="log1p requires non-negative values.",
        )
        self._shape = array.shape[1]
        return self

    def transform(self, data):
        check_fitted(self, "_shape")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != self._shape:
            raise BadInputError(
                "Input feature count does not match the fitted transformer."
            )
        _validate_local_domain(
            array,
            predicate=lambda x: x >= 0,
            message="log1p requires non-negative values.",
        )
        result = np.log1p(array)
        return _restore_shape(result, squeezed)


class SqrtTransformer(_BaseTransformer):
    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _validate_domain(
            array,
            agg_client,
            predicate=lambda x: x >= 0,
            message="sqrt requires non-negative values.",
        )
        self._shape = array.shape[1]
        return self

    def transform(self, data):
        check_fitted(self, "_shape")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != self._shape:
            raise BadInputError(
                "Input feature count does not match the fitted transformer."
            )
        _validate_local_domain(
            array,
            predicate=lambda x: x >= 0,
            message="sqrt requires non-negative values.",
        )
        return _restore_shape(np.sqrt(array), squeezed)


class AsinhTransformer(_BaseTransformer):
    def fit(self, data, *, agg_client=None):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        self._shape = array.shape[1]
        return self

    def transform(self, data):
        check_fitted(self, "_shape")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != self._shape:
            raise BadInputError(
                "Input feature count does not match the fitted transformer."
            )
        return _restore_shape(np.arcsinh(array), squeezed)


class StandardScaler(_BaseTransformer):
    @classmethod
    def from_recipe(cls, recipe):
        recipe = ScalerRecipe.model_validate(recipe)
        if recipe.primitive_name != "standard_scaler":
            raise BadInputError("Recipe does not belong to StandardScaler.")
        scaler = cls()
        scaler.mean_ = np.asarray(recipe.location, dtype=float)
        scaler.scale_ = np.asarray(recipe.scale, dtype=float)
        scaler.recipe_ = recipe
        return scaler

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        self.mean_, self.scale_ = _global_population_scale(array, agg_client)
        self.recipe_ = ScalerRecipe(
            primitive_name="standard_scaler",
            location=self.mean_.tolist(),
            scale=self.scale_.tolist(),
        )
        return self

    def transform(self, data):
        check_fitted(self, "mean_", "scale_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.mean_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        return _restore_shape((array - self.mean_) / self.scale_, squeezed)


class RobustScalerApprox(_BaseTransformer):
    @classmethod
    def from_recipe(cls, recipe):
        recipe = ScalerRecipe.model_validate(recipe)
        if recipe.primitive_name != "robust_scaler":
            raise BadInputError("Recipe does not belong to RobustScalerApprox.")
        scaler = cls()
        scaler.center_ = np.asarray(recipe.location, dtype=float)
        scaler.scale_ = np.asarray(recipe.scale, dtype=float)
        scaler.recipe_ = recipe
        return scaler

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        q25, self.center_, q75 = _global_quartiles(array, agg_client)
        self.scale_ = np.where(q75 - q25 == 0, 1.0, q75 - q25)
        self.recipe_ = ScalerRecipe(
            primitive_name="robust_scaler",
            location=self.center_.tolist(),
            scale=self.scale_.tolist(),
        )
        return self

    def transform(self, data):
        check_fitted(self, "center_", "scale_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.center_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        return _restore_shape((array - self.center_) / self.scale_, squeezed)


class MinMaxScaler(_BaseTransformer):
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        check_interval(feature_range)
        self.feature_range = tuple(float(value) for value in feature_range)

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        self.data_min_, self.data_max_ = _global_min_max(array, agg_client)
        self._scale = np.full_like(
            self.data_min_, self.feature_range[1] - self.feature_range[0], dtype=float
        )
        non_constant = self.data_max_ != self.data_min_
        self._scale[non_constant] = (self.feature_range[1] - self.feature_range[0]) / (
            self.data_max_[non_constant] - self.data_min_[non_constant]
        )
        self.recipe_ = ScalerRecipe(
            primitive_name="minmax_scaler",
            location=self.data_min_.tolist(),
            scale=self._scale.tolist(),
            data_min=self.data_min_.tolist(),
            data_max=self.data_max_.tolist(),
            feature_range=self.feature_range,
        )
        return self

    @classmethod
    def from_recipe(cls, recipe):
        recipe = ScalerRecipe.model_validate(recipe)
        if recipe.primitive_name != "minmax_scaler":
            raise BadInputError("Recipe does not belong to MinMaxScaler.")
        if recipe.data_min is None or recipe.data_max is None:
            raise BadInputError("MinMaxScaler recipe is missing data bounds.")
        scaler = cls(feature_range=recipe.feature_range or (0.0, 1.0))
        scaler.data_min_ = np.asarray(recipe.data_min, dtype=float)
        scaler.data_max_ = np.asarray(recipe.data_max, dtype=float)
        scaler._scale = np.asarray(recipe.scale, dtype=float)
        scaler.recipe_ = recipe
        return scaler

    def transform(self, data):
        check_fitted(self, "data_min_", "data_max_", "_scale")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.data_min_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        result = (array - self.data_min_) * self._scale + self.feature_range[0]
        return _restore_shape(result, squeezed)


class MaxAbsScaler(_BaseTransformer):
    @classmethod
    def from_recipe(cls, recipe):
        recipe = ScalerRecipe.model_validate(recipe)
        if recipe.primitive_name != "maxabs_scaler":
            raise BadInputError("Recipe does not belong to MaxAbsScaler.")
        scaler = cls()
        scaler.scale_ = np.asarray(recipe.scale, dtype=float)
        scaler.max_abs_ = scaler.scale_.copy()
        scaler.recipe_ = recipe
        return scaler

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        valid = ~np.isnan(array)
        local_max = np.max(np.where(valid, np.abs(array), 0.0), axis=0)
        self.max_abs_ = np.asarray(agg_client.max(local_max), dtype=float)
        local_count = valid.sum(axis=0).astype(float)
        total_count = np.asarray(agg_client.sum(local_count), dtype=float)
        if np.any(total_count <= 0) or np.any(~np.isfinite(self.max_abs_)):
            raise BadInputError(
                "MaxAbsScaler cannot fit a feature without valid values."
            )
        self.scale_ = np.where(self.max_abs_ == 0, 1.0, self.max_abs_)
        self.recipe_ = ScalerRecipe(
            primitive_name="maxabs_scaler",
            location=[0.0] * len(self.scale_),
            scale=self.scale_.tolist(),
        )
        return self

    def transform(self, data):
        check_fitted(self, "scale_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.scale_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        return _restore_shape(array / self.scale_, squeezed)


class Winsorizer(_BaseTransformer):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        check_scalar(lower_quantile, name="lower_quantile", minimum=0, maximum=1)
        check_scalar(upper_quantile, name="upper_quantile", minimum=0, maximum=1)
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise BadInputError(
                "Winsorizer quantiles must satisfy 0 <= lower < upper <= 1."
            )
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)

    @classmethod
    def from_recipe(cls, recipe):
        recipe = WinsorizerRecipe.model_validate(recipe)
        winsorizer = cls(recipe.lower_quantile, recipe.upper_quantile)
        winsorizer.lower_bounds_ = np.asarray(recipe.lower_bounds, dtype=float)
        winsorizer.upper_bounds_ = np.asarray(recipe.upper_bounds, dtype=float)
        winsorizer.recipe_ = recipe
        return winsorizer

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        self.lower_bounds_ = np.asarray(
            [
                _percentile(array[:, i], self.lower_quantile, agg_client)
                for i in range(array.shape[1])
            ]
        )
        self.upper_bounds_ = np.asarray(
            [
                _percentile(array[:, i], self.upper_quantile, agg_client)
                for i in range(array.shape[1])
            ]
        )
        self.recipe_ = WinsorizerRecipe(
            lower_bounds=self.lower_bounds_.tolist(),
            upper_bounds=self.upper_bounds_.tolist(),
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile,
        )
        return self

    def transform(self, data):
        check_fitted(self, "lower_bounds_", "upper_bounds_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.lower_bounds_):
            raise BadInputError(
                "Input feature count does not match the fitted winsorizer."
            )
        return _restore_shape(
            np.clip(array, self.lower_bounds_, self.upper_bounds_), squeezed
        )


class OutlierDetector(_BaseTransformer):
    def __init__(self, mode: str = "zscore", threshold: float = 3.0):
        check_choice(mode, ("zscore", "robust"), name="mode")
        check_scalar(
            threshold, name="threshold", minimum=0, include_boundaries="neither"
        )
        self.mode = mode
        self.threshold = float(threshold)

    @classmethod
    def from_recipe(cls, recipe):
        recipe = OutlierDetectorRecipe.model_validate(recipe)
        detector = cls(mode=recipe.mode, threshold=recipe.threshold)
        parameters = np.asarray(recipe.parameters_by_feature, dtype=float)
        detector.location_ = parameters[:, 0]
        detector.scale_ = parameters[:, 1]
        detector.recipe_ = recipe
        return detector

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        if self.mode == "zscore":
            self.location_, self.scale_ = _global_population_scale(array, agg_client)
            parameters = np.column_stack((self.location_, self.scale_)).tolist()
        else:
            q25, self.location_, q75 = _global_quartiles(array, agg_client)
            self.scale_ = np.where(q75 - q25 == 0, 1.0, q75 - q25)
            parameters = np.column_stack((self.location_, self.scale_)).tolist()
        self.recipe_ = OutlierDetectorRecipe(
            mode=self.mode,
            parameters_by_feature=parameters,
            threshold=self.threshold,
        )
        return self

    def transform(self, data):
        check_fitted(self, "location_", "scale_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.location_):
            raise BadInputError(
                "Input feature count does not match the fitted detector."
            )
        if self.mode == "zscore":
            mask = np.abs((array - self.location_) / self.scale_) > self.threshold
        else:
            mask = np.abs(array - self.location_) > self.threshold * self.scale_
        return _restore_shape(np.where(np.isnan(array), False, mask), squeezed)


def _yeo_johnson(values: np.ndarray, lambd: float) -> np.ndarray:
    if lambd == 1:
        return values.copy()
    result = np.empty_like(values, dtype=float)
    nonnegative = values >= 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if lambd == 0:
            result[nonnegative] = np.log1p(values[nonnegative])
        else:
            result[nonnegative] = (
                np.expm1(lambd * np.log1p(values[nonnegative])) / lambd
            )
        if lambd == 2:
            result[~nonnegative] = -np.log1p(-values[~nonnegative])
        else:
            result[~nonnegative] = -np.expm1(
                (2 - lambd) * np.log1p(-values[~nonnegative])
            ) / (2 - lambd)
    return result


def _yeo_johnson_profile_values(values, lambd, minimum, maximum):
    """Remove only a known additive constant when evaluating variance.

    On one-sided data, saturated power branches contain an additive constant
    that dwarfs the remaining variation. Omitting it leaves variance and the
    profile likelihood unchanged while retaining floating-point information.
    This does not shift the input or alter the actual transform/replay output.
    """
    if minimum >= 0 and lambd * np.log1p(minimum) < -1:
        return np.exp(lambd * np.log1p(values)) / lambd
    power = 2 - lambd
    if maximum <= 0 and power * np.log1p(-maximum) < -1:
        return -np.exp(power * np.log1p(-values)) / power
    return _yeo_johnson(values, lambd)


class YeoJohnsonTransformer(_BaseTransformer):
    """Globally fit a budgeted adaptive power transformation per feature.

    Search starts near [-5, 5], expands when the best candidate is an endpoint,
    and stays inside global float64 exponent bounds. The budget covers all
    objective evaluations per feature, including expansion. Approximate fits
    retain the best finite candidate and explicitly record the stop reason.
    All participants must use identical settings and ordered feature layout.
    """

    def __init__(
        self, *, max_evaluations=64, lambda_tolerance=1e-5, likelihood_tolerance=1e-10
    ):
        if isinstance(max_evaluations, (bool, np.bool_)):
            raise BadInputError("max_evaluations must be an integer.")
        check_scalar(
            max_evaluations, name="max_evaluations", target_type=Integral, minimum=1
        )
        check_scalar(
            lambda_tolerance,
            name="lambda_tolerance",
            minimum=0,
            include_boundaries="neither",
        )
        check_scalar(likelihood_tolerance, name="likelihood_tolerance", minimum=0)
        self.max_evaluations = int(max_evaluations)
        self.lambda_tolerance = float(lambda_tolerance)
        self.likelihood_tolerance = float(likelihood_tolerance)

    @classmethod
    def from_recipe(cls, recipe):
        recipe = YeoJohnsonRecipe.model_validate(recipe)
        transformer = cls(
            max_evaluations=recipe.max_evaluations,
            lambda_tolerance=recipe.lambda_tolerance,
            likelihood_tolerance=recipe.likelihood_tolerance,
        )
        transformer.lambdas_ = np.asarray(recipe.lambdas, dtype=float)
        transformer.fit_diagnostics_ = recipe.fit_diagnostics
        transformer.recipe_ = recipe
        return transformer

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        check_common_configuration(
            [
                array.shape[1],
                self.max_evaluations,
                self.lambda_tolerance,
                self.likelihood_tolerance,
            ],
            agg_client,
        )
        global_min, global_max = _global_min_max(array, agg_client)
        lambdas = []
        diagnostics = []
        for idx in range(array.shape[1]):
            values = array[:, idx]
            valid = values[~np.isnan(values)]
            if global_min[idx] == global_max[idx]:
                lambdas.append(1.0)
                diagnostics.append(
                    {
                        "status": "constant",
                        "reason": "constant_feature",
                        "evaluations": 0,
                        "searched_bounds": (1.0, 1.0),
                        "safe_bounds": (1.0, 1.0),
                        "objective": None,
                    }
                )
                continue

            def objective(lambd):
                # These shared-extrema checks take the same branch everywhere,
                # before feature-specific aggregation. A collapsed output is
                # not a useful fitted transform even if its ideal LL is finite.
                try:
                    endpoints = _yeo_johnson(
                        np.array([global_min[idx], global_max[idx]]), lambd
                    )
                except FloatingPointError:
                    return np.inf
                if endpoints[0] == endpoints[1]:
                    return np.inf
                local_error = False
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        transformed = _yeo_johnson_profile_values(
                            valid, float(lambd), global_min[idx], global_max[idx]
                        )
                        local_sum = transformed.sum()
                except FloatingPointError:
                    local_error = True
                    local_sum = 0.0
                total_sum, total_error = np.asarray(
                    agg_client.sum([local_sum, float(local_error)]), dtype=float
                )
                if total_error or not np.isfinite(total_sum):
                    return np.inf
                mean = total_sum / total_n
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        differences = transformed - mean
                        local_squared = np.square(differences).sum()
                        local_correction = differences.sum()
                except FloatingPointError:
                    local_error = True
                    local_squared = 0.0
                    local_correction = 0.0
                total_squared, correction, total_error = np.asarray(
                    agg_client.sum(
                        [local_squared, local_correction, float(local_error)]
                    ),
                    dtype=float,
                )
                if total_error or not np.isfinite(total_squared):
                    return np.inf
                variance = (
                    total_squared - correction * (correction / total_n)
                ) / total_n
                if variance <= 0 or not np.isfinite(variance):
                    return np.inf
                return 0.5 * np.log(variance) - (lambd - 1) * (total_jacobian / total_n)

            jacobian = np.sign(valid) * np.log1p(np.abs(valid))
            total_n, total_jacobian = np.asarray(
                agg_client.sum([len(valid), jacobian.sum()]), dtype=float
            )
            bounds = safe_lambda_bounds(global_min[idx], global_max[idx], total_n)
            lambd, diagnostic = adaptive_lambda_search(
                objective,
                bounds,
                max_evaluations=self.max_evaluations,
                lambda_tolerance=self.lambda_tolerance,
                likelihood_tolerance=self.likelihood_tolerance,
            )
            lambdas.append(lambd)
            diagnostics.append(diagnostic)
        self.lambdas_ = np.asarray(lambdas)
        self.recipe_ = YeoJohnsonRecipe(
            lambdas=self.lambdas_.tolist(),
            max_evaluations=self.max_evaluations,
            lambda_tolerance=self.lambda_tolerance,
            likelihood_tolerance=self.likelihood_tolerance,
            fit_diagnostics=diagnostics,
        )
        self.fit_diagnostics_ = self.recipe_.fit_diagnostics
        return self

    def transform(self, data):
        check_fitted(self, "lambdas_")
        array, squeezed = _as_2d_numeric(data)
        _reject_infinite(array)
        if array.shape[1] != len(self.lambdas_):
            raise BadInputError(
                "Input feature count does not match the fitted transformer."
            )
        result = np.empty_like(array, dtype=float)
        for idx, lambd in enumerate(self.lambdas_):
            valid = ~np.isnan(array[:, idx])
            result[:, idx] = np.nan
            try:
                result[valid, idx] = _yeo_johnson(array[valid, idx], float(lambd))
            except FloatingPointError:
                raise BadInputError(
                    "Yeo-Johnson transformation is undefined for the input."
                ) from None
        return _restore_shape(result, squeezed)


class QuantileTransformerApprox(_BaseTransformer):
    """Map values through approximate global quantiles.

    The trusted caller supplies the platform privacy minimum. Fitting requires
    at least max(platform_minimum, min_observations_per_interval) observations
    per requested rank interval. This is a cohort-size guard, not differential
    privacy or a guarantee on the occupancy of value bins.

    Quantiles share batched histogram refinement, with two additional rounds by
    default. Selected bins are enlarged to meet the effective support floor;
    only supported groups expose aggregate extrema. Recipes retain achieved
    uncertainty intervals instead of claiming a fixed accuracy. Exact global
    endpoints are retained separately from the interior support rule.
    """

    def __init__(
        self,
        n_quantiles: int = 100,
        output_distribution: str = "uniform",
        min_observations_per_interval: int = 5,
        *,
        platform_minimum: int,
        n_bins: int = 256,
        max_refinement_rounds: int = 2,
    ):
        for name, value, minimum in (
            ("n_quantiles", n_quantiles, 2),
            ("min_observations_per_interval", min_observations_per_interval, 1),
            ("platform_minimum", platform_minimum, 1),
            ("n_bins", n_bins, 2),
            ("max_refinement_rounds", max_refinement_rounds, 0),
        ):
            if isinstance(value, (bool, np.bool_)):
                raise BadInputError(f"{name} must be an integer.")
            check_scalar(value, name=name, target_type=Integral, minimum=minimum)
        check_choice(
            output_distribution, ("uniform", "normal"), name="output_distribution"
        )
        self.n_quantiles = int(n_quantiles)
        self.output_distribution = output_distribution
        self.min_observations_per_interval = int(min_observations_per_interval)
        self.platform_minimum = int(platform_minimum)
        self.n_bins = int(n_bins)
        self.max_refinement_rounds = int(max_refinement_rounds)

    @classmethod
    def from_recipe(cls, recipe):
        recipe = QuantileTransformerRecipe.model_validate(recipe)
        transformer = cls(
            n_quantiles=len(recipe.references),
            output_distribution=recipe.output_distribution,
            min_observations_per_interval=recipe.min_observations_per_interval,
            platform_minimum=recipe.platform_minimum,
            n_bins=recipe.n_bins,
            max_refinement_rounds=recipe.max_refinement_rounds,
        )
        transformer.quantiles_ = np.asarray(recipe.quantiles, dtype=float)
        transformer.references_ = np.asarray(recipe.references, dtype=float)
        transformer.fit_diagnostics_ = recipe.fit_diagnostics
        transformer.recipe_ = recipe
        return transformer

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        check_common_configuration(
            [
                array.shape[1],
                self.n_quantiles,
                self.output_distribution,
                self.n_bins,
                self.max_refinement_rounds,
            ],
            agg_client,
        )
        platform_minimum, configured_minimum = np.asarray(
            agg_client.max([self.platform_minimum, self.min_observations_per_interval]),
            dtype=float,
        )
        effective_minimum = int(max(platform_minimum, configured_minimum))
        global_count = np.asarray(
            agg_client.sum((~np.isnan(array)).sum(axis=0).astype(float)), dtype=float
        )
        required_count = effective_minimum * (self.n_quantiles - 1)
        if np.any(global_count < required_count):
            raise BadInputError(
                "QuantileTransformer requires more global observations per interval."
            )
        references = np.linspace(0.0, 1.0, self.n_quantiles)
        minimum, maximum = _global_min_max(array, agg_client)
        quantiles = []
        diagnostics = []
        for idx in range(array.shape[1]):
            values, diagnostic = batched_quantiles(
                array[:, idx],
                references,
                minimum=minimum[idx],
                maximum=maximum[idx],
                count=int(global_count[idx]),
                minimum_support=effective_minimum,
                n_bins=self.n_bins,
                max_refinement_rounds=self.max_refinement_rounds,
                agg_client=agg_client,
            )
            quantiles.append(values)
            diagnostics.append(diagnostic)
        self.quantiles_ = np.asarray(quantiles, dtype=float)
        self.references_ = references
        self.recipe_ = QuantileTransformerRecipe(
            quantiles=self.quantiles_.tolist(),
            references=references.tolist(),
            output_distribution=self.output_distribution,
            min_observations_per_interval=effective_minimum,
            platform_minimum=int(platform_minimum),
            n_bins=self.n_bins,
            max_refinement_rounds=self.max_refinement_rounds,
            fit_diagnostics=diagnostics,
        )
        self.fit_diagnostics_ = self.recipe_.fit_diagnostics
        return self

    def transform(self, data):
        check_fitted(self, "quantiles_", "references_")
        array, squeezed = _as_2d_numeric(data)
        _reject_infinite(array)
        if array.shape[1] != self.quantiles_.shape[0]:
            raise BadInputError(
                "Input feature count does not match the fitted transformer."
            )
        result = np.empty_like(array, dtype=float)
        for idx in range(array.shape[1]):
            if np.all(self.quantiles_[idx] == self.quantiles_[idx, 0]):
                constant_value = 0.5 if self.output_distribution == "uniform" else 0.0
                result[:, idx] = np.where(
                    np.isnan(array[:, idx]), np.nan, constant_value
                )
                continue
            unique_quantiles, inverse = np.unique(
                self.quantiles_[idx], return_inverse=True
            )
            reference_sum = np.bincount(
                inverse, weights=self.references_, minlength=len(unique_quantiles)
            )
            reference_count = np.bincount(inverse, minlength=len(unique_quantiles))
            unique_references = reference_sum / reference_count
            mapped = np.interp(array[:, idx], unique_quantiles, unique_references)
            if self.output_distribution == "normal":
                mapped = ndtri(np.clip(mapped, 1e-7, 1 - 1e-7))
            result[:, idx] = mapped
            result[np.isnan(array[:, idx]), idx] = np.nan
        return _restore_shape(result, squeezed)
