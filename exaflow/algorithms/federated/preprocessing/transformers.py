"""Federated mathematical preprocessing primitives.

The classes in this module operate on local NumPy partitions.  Fitted
parameters are obtained through the existing AggregationClient operations;
raw observations never leave a worker.
"""

from __future__ import annotations

from numbers import Number
from typing import Any
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import ndtri

from exaflow.algorithms.federated.preprocessing.recipes import OutlierDetectorRecipe
from exaflow.algorithms.federated.preprocessing.recipes import QuantileTransformerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import ScalerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import SimpleImputerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import WinsorizerRecipe
from exaflow.algorithms.federated.preprocessing.recipes import YeoJohnsonRecipe
from exaflow.algorithms.federated.statistics.percentile import Percentile
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


def _as_2d_numeric(data: Any, *, name: str = "Input") -> tuple[np.ndarray, bool]:
    try:
        values = data.to_numpy(copy=True) if isinstance(data, pd.DataFrame) else data
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise BadInputError(f"{name} must contain numeric values.") from None
    if array.ndim == 1:
        return array.reshape(-1, 1).copy(), True
    if array.ndim != 2:
        raise BadInputError(f"{name} must be a one- or two-dimensional array.")
    return array.copy(), False


def _restore_shape(array: np.ndarray, squeezed: bool) -> np.ndarray:
    return array[:, 0] if squeezed else array


def _require_rows(array: np.ndarray) -> None:
    if array.shape[0] == 0:
        raise BadInputError("Input must contain at least one observation.")


def _reject_infinite(array: np.ndarray) -> None:
    if np.isinf(array).any():
        raise BadInputError("Input must not contain infinite values.")


def _require_fitted(transformer: object, *attributes: str) -> None:
    if any(not hasattr(transformer, attribute) for attribute in attributes):
        raise BadInputError("Transformer is not fitted.")


def _validate_local_domain(array: np.ndarray, *, predicate, message: str) -> None:
    invalid = np.isinf(array) | (~np.isnan(array) & ~predicate(array))
    if invalid.any():
        raise BadInputError(message)


def _stabilize_scale(raw_scale: np.ndarray, location: np.ndarray) -> np.ndarray:
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(location))
    return np.where(raw_scale <= tolerance, 1.0, raw_scale)


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
    valid = ~np.isnan(array)
    local_min = np.min(np.where(valid, array, np.inf), axis=0)
    local_max = np.max(np.where(valid, array, -np.inf), axis=0)
    minimum = np.asarray(agg_client.min(local_min), dtype=float)
    maximum = np.asarray(agg_client.max(local_max), dtype=float)
    if np.any(~np.isfinite(minimum)) or np.any(~np.isfinite(maximum)):
        raise BadInputError("At least one feature has no valid observations.")
    return minimum, maximum


def _global_population_scale(
    array: np.ndarray, agg_client: AggregationClient
) -> tuple[np.ndarray, np.ndarray]:
    mean = _global_mean(array, agg_client)
    valid = ~np.isnan(array)
    local_squared = np.nansum(np.where(valid, (array - mean) ** 2, 0.0), axis=0)
    local_count = valid.sum(axis=0).astype(float)
    total_squared = np.asarray(agg_client.sum(local_squared), dtype=float)
    total_count = np.asarray(agg_client.sum(local_count), dtype=float)
    if np.any(total_count <= 0):
        raise BadInputError("At least one feature has no valid observations.")
    variance = np.maximum(total_squared / total_count, 0.0)
    return mean, np.sqrt(variance)


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


class _BaseTransformer:
    recipe_: Any = None

    def fit_transform(self, data, *, agg_client):
        return self.fit(data, agg_client=agg_client).transform(data)


class SimpleImputer(_BaseTransformer):
    """Impute missing values using globally fitted statistics.

    Constant imputation defaults to ``0`` for numeric input when no fill value
    is supplied.
    """

    def __init__(self, strategy: str = "mean", fill_value: Any = None):
        if strategy not in {"mean", "median", "constant", "most_frequent"}:
            raise BadInputError("Unsupported imputation strategy.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: Optional[np.ndarray] = None

    @classmethod
    def from_recipe(cls, recipe):
        recipe = SimpleImputerRecipe.model_validate(recipe)
        imputer = cls(strategy=recipe.strategy)
        imputer.statistics_ = np.asarray(recipe.statistics, dtype=object)
        imputer.recipe_ = recipe
        return imputer

    def fit(self, data, *, agg_client):
        if self.strategy == "constant":
            values = np.asarray(data, dtype=object)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.ndim != 2:
                raise BadInputError("Input must be one- or two-dimensional.")
            if self.fill_value is None:
                try:
                    _as_2d_numeric(data)
                except BadInputError:
                    raise BadInputError(
                        "fill_value is required for non-numeric constant imputation."
                    ) from None
                self.fill_value = 0
            self.statistics_ = np.asarray(
                [self.fill_value] * values.shape[1], dtype=object
            )
        elif self.strategy in {"mean", "median"}:
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
            values = np.asarray(data, dtype=object)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.ndim != 2 or values.shape[0] == 0:
                raise BadInputError("Input must contain at least one observation.")
            statistics = []
            for idx in range(values.shape[1]):
                column = values[:, idx]
                observed = [value for value in column if not pd.isna(value)]
                if any(
                    isinstance(value, Number)
                    and not isinstance(value, (bool, np.bool_))
                    for value in observed
                ):
                    raise BadInputError(
                        "numeric mode requires a safe histogram protocol."
                    )
                categories = sorted(
                    set(observed), key=lambda value: (type(value).__name__, str(value))
                )
                if not categories:
                    categories = []
                global_categories = list(agg_client.union(categories))
                if not global_categories:
                    raise BadInputError(
                        "At least one feature has no valid observations."
                    )
                counts = np.asarray(
                    agg_client.sum(
                        np.asarray(
                            [
                                observed.count(category)
                                for category in global_categories
                            ],
                            dtype=float,
                        )
                    ),
                    dtype=float,
                )
                statistics.append(global_categories[int(np.argmax(counts))])
            self.statistics_ = np.asarray(statistics, dtype=object)
        self.recipe_ = SimpleImputerRecipe(
            strategy=self.strategy,
            statistics=self.statistics_.tolist(),
        )
        return self

    def transform(self, data):
        _require_fitted(self, "statistics_")
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
        for idx, statistic in enumerate(self.statistics_):
            missing = pd.isna(values[:, idx])
            values[missing, idx] = statistic
        if self.strategy in {"mean", "median"}:
            values = values.astype(float)
        return values[:, 0] if squeezed else values


class MissingIndicator(_BaseTransformer):
    def fit(self, data, *, agg_client=None):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        self._n_features = array.shape[1]
        self._squeezed = squeezed
        self.recipe_ = {"primitive_name": "missing_indicator", "schema_version": "1"}
        return self

    def transform(self, data):
        _require_fitted(self, "_n_features")
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
        self._squeezed = squeezed
        self.recipe_ = {"primitive_name": "log1p", "schema_version": "1"}
        return self

    def transform(self, data):
        _require_fitted(self, "_shape")
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
        self._squeezed = squeezed
        self.recipe_ = {"primitive_name": "sqrt", "schema_version": "1"}
        return self

    def transform(self, data):
        _require_fitted(self, "_shape")
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
        self.recipe_ = {"primitive_name": "asinh", "schema_version": "1"}
        return self

    def transform(self, data):
        _require_fitted(self, "_shape")
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
        self.mean_, raw_scale = _global_population_scale(array, agg_client)
        self.scale_ = _stabilize_scale(raw_scale, self.mean_)
        self._squeezed = squeezed
        self.recipe_ = ScalerRecipe(
            primitive_name="standard_scaler",
            location=self.mean_.tolist(),
            scale=self.scale_.tolist(),
        )
        return self

    def transform(self, data):
        _require_fitted(self, "mean_", "scale_")
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
        self.center_ = np.asarray(
            [_percentile(array[:, i], 0.5, agg_client) for i in range(array.shape[1])]
        )
        q25 = np.asarray(
            [_percentile(array[:, i], 0.25, agg_client) for i in range(array.shape[1])]
        )
        q75 = np.asarray(
            [_percentile(array[:, i], 0.75, agg_client) for i in range(array.shape[1])]
        )
        self.scale_ = np.where(q75 - q25 == 0, 1.0, q75 - q25)
        self._squeezed = squeezed
        self.recipe_ = ScalerRecipe(
            primitive_name="robust_scaler",
            location=self.center_.tolist(),
            scale=self.scale_.tolist(),
        )
        return self

    def transform(self, data):
        _require_fitted(self, "center_", "scale_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.center_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        return _restore_shape((array - self.center_) / self.scale_, squeezed)


class MinMaxScaler(_BaseTransformer):
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        if len(feature_range) != 2 or not np.isfinite(feature_range).all():
            raise BadInputError("feature_range must contain two finite values.")
        if feature_range[0] >= feature_range[1]:
            raise BadInputError("feature_range minimum must be less than its maximum.")
        self.feature_range = tuple(float(value) for value in feature_range)

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        self.data_min_, self.data_max_ = _global_min_max(array, agg_client)
        self._scale = np.ones_like(self.data_min_, dtype=float)
        non_constant = self.data_max_ != self.data_min_
        self._scale[non_constant] = (self.feature_range[1] - self.feature_range[0]) / (
            self.data_max_[non_constant] - self.data_min_[non_constant]
        )
        self._squeezed = squeezed
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
        _require_fitted(self, "data_min_", "data_max_", "_scale")
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
        self._squeezed = squeezed
        self.recipe_ = ScalerRecipe(
            primitive_name="maxabs_scaler",
            location=[0.0] * len(self.scale_),
            scale=self.scale_.tolist(),
        )
        return self

    def transform(self, data):
        _require_fitted(self, "scale_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.scale_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        return _restore_shape(array / self.scale_, squeezed)


class Winsorizer(_BaseTransformer):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
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
        self._squeezed = squeezed
        return self

    def transform(self, data):
        _require_fitted(self, "lower_bounds_", "upper_bounds_")
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
        if mode not in {"zscore", "robust"}:
            raise BadInputError("OutlierDetector mode must be 'zscore' or 'robust'.")
        if not np.isfinite(threshold) or threshold <= 0:
            raise BadInputError(
                "OutlierDetector threshold must be positive and finite."
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
            self.scale_ = _stabilize_scale(self.scale_, self.location_)
            parameters = np.column_stack((self.location_, self.scale_)).tolist()
        else:
            self.location_ = np.asarray(
                [
                    _percentile(array[:, i], 0.5, agg_client)
                    for i in range(array.shape[1])
                ]
            )
            q25 = np.asarray(
                [
                    _percentile(array[:, i], 0.25, agg_client)
                    for i in range(array.shape[1])
                ]
            )
            q75 = np.asarray(
                [
                    _percentile(array[:, i], 0.75, agg_client)
                    for i in range(array.shape[1])
                ]
            )
            self.scale_ = np.where(q75 - q25 == 0, 1.0, q75 - q25)
            parameters = np.column_stack((self.location_, self.scale_)).tolist()
        self._squeezed = squeezed
        self.recipe_ = OutlierDetectorRecipe(
            mode=self.mode,
            parameters_by_feature=parameters,
            threshold=self.threshold,
        )
        return self

    def transform(self, data):
        _require_fitted(self, "location_", "scale_")
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
    result = np.empty_like(values, dtype=float)
    nonnegative = values >= 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if abs(lambd) < 1e-12:
            result[nonnegative] = np.log1p(values[nonnegative])
        else:
            result[nonnegative] = ((values[nonnegative] + 1) ** lambd - 1) / lambd
        if abs(lambd - 2) < 1e-12:
            result[~nonnegative] = -np.log1p(-values[~nonnegative])
        else:
            result[~nonnegative] = -(
                ((-values[~nonnegative] + 1) ** (2 - lambd) - 1) / (2 - lambd)
            )
    return result


class YeoJohnsonTransformer(_BaseTransformer):
    @classmethod
    def from_recipe(cls, recipe):
        recipe = YeoJohnsonRecipe.model_validate(recipe)
        transformer = cls()
        transformer.lambdas_ = np.asarray(recipe.lambdas, dtype=float)
        transformer.recipe_ = recipe
        return transformer

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        global_min, global_max = _global_min_max(array, agg_client)
        lambdas = []
        for idx in range(array.shape[1]):
            values = array[:, idx]
            valid = values[~np.isnan(values)]
            if global_min[idx] == global_max[idx]:
                lambdas.append(1.0)
                continue

            def objective(lambd):
                transformed = _yeo_johnson(valid, float(lambd))
                total_sum, total_count = _global_sum_count(
                    transformed.reshape(-1, 1), agg_client
                )
                if total_count[0] <= 0:
                    return np.inf
                mean = total_sum[0] / total_count[0]
                local_squared = np.square(transformed - mean).sum()
                total_squared = float(
                    np.asarray(agg_client.sum([local_squared]), dtype=float)[0]
                )
                variance = total_squared / total_count[0]
                if variance <= 0 or not np.isfinite(variance):
                    return np.inf
                return (
                    total_count[0] / 2 * np.log(variance) - (lambd - 1) * total_jacobian
                )

            jacobian = np.sign(valid) * np.log1p(np.abs(valid))
            total_jacobian = float(
                np.asarray(agg_client.sum([jacobian.sum()]), dtype=float)[0]
            )
            result = minimize_scalar(objective, bounds=(-5, 5), method="bounded")
            if not result.success or not np.isfinite(result.x):
                raise BadInputError("Yeo-Johnson lambda fitting did not converge.")
            lambdas.append(float(result.x))
        self.lambdas_ = np.asarray(lambdas)
        self._squeezed = squeezed
        self.recipe_ = YeoJohnsonRecipe(lambdas=self.lambdas_.tolist())
        return self

    def transform(self, data):
        _require_fitted(self, "lambdas_")
        array, squeezed = _as_2d_numeric(data)
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

    By default, fitting requires five global observations per quantile interval
    to avoid exposing fitted order statistics from very small cohorts.
    """

    def __init__(
        self,
        n_quantiles: int = 100,
        output_distribution: str = "uniform",
        min_observations_per_interval: int = 5,
    ):
        if n_quantiles < 2 or int(n_quantiles) != n_quantiles:
            raise BadInputError(
                "n_quantiles must be an integer greater than or equal to 2."
            )
        if output_distribution not in {"uniform", "normal"}:
            raise BadInputError("output_distribution must be 'uniform' or 'normal'.")
        if (
            int(min_observations_per_interval) != min_observations_per_interval
            or min_observations_per_interval < 1
        ):
            raise BadInputError(
                "min_observations_per_interval must be a positive integer."
            )
        self.n_quantiles = int(n_quantiles)
        self.output_distribution = output_distribution
        self.min_observations_per_interval = int(min_observations_per_interval)

    @classmethod
    def from_recipe(cls, recipe):
        recipe = QuantileTransformerRecipe.model_validate(recipe)
        transformer = cls(
            n_quantiles=len(recipe.references),
            output_distribution=recipe.output_distribution,
            min_observations_per_interval=recipe.min_observations_per_interval,
        )
        transformer.quantiles_ = np.asarray(recipe.quantiles, dtype=float)
        transformer.references_ = np.asarray(recipe.references, dtype=float)
        transformer.recipe_ = recipe
        return transformer

    def fit(self, data, *, agg_client):
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        global_count = np.asarray(
            agg_client.sum((~np.isnan(array)).sum(axis=0).astype(float)), dtype=float
        )
        required_count = self.min_observations_per_interval * (self.n_quantiles - 1)
        if np.any(global_count < required_count):
            raise BadInputError(
                "QuantileTransformer requires more global observations per interval."
            )
        references = np.linspace(0.0, 1.0, self.n_quantiles)
        quantiles = []
        for idx in range(array.shape[1]):
            quantiles.append(
                [_percentile(array[:, idx], q, agg_client) for q in references]
            )
        self.quantiles_ = np.maximum.accumulate(
            np.asarray(quantiles, dtype=float), axis=1
        )
        self.references_ = references
        self._squeezed = squeezed
        self.recipe_ = QuantileTransformerRecipe(
            quantiles=self.quantiles_.tolist(),
            references=references.tolist(),
            output_distribution=self.output_distribution,
            min_observations_per_interval=self.min_observations_per_interval,
        )
        return self

    def transform(self, data):
        _require_fitted(self, "quantiles_", "references_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != self.quantiles_.shape[0]:
            raise BadInputError(
                "Input feature count does not match the fitted transformer."
            )
        result = np.empty_like(array, dtype=float)
        for idx in range(array.shape[1]):
            if np.allclose(self.quantiles_[idx], self.quantiles_[idx, 0]):
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
