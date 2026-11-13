"""MinMaxScaler: federated fit and named-column transformation."""

from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.feature_engineering.numerical import _as_2d_numeric
from exaflow.algorithms.federated.feature_engineering.numerical import _global_min_max
from exaflow.algorithms.federated.feature_engineering.numerical import _reject_infinite
from exaflow.algorithms.federated.feature_engineering.numerical import _require_rows
from exaflow.algorithms.federated.feature_engineering.numerical import _restore_shape
from exaflow.algorithms.federated.feature_engineering.scaler_state import ScalerRecipe
from exaflow.algorithms.federated.feature_engineering.state import load_state
from exaflow.algorithms.federated.feature_engineering.transformer import (
    FeatureTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import check_fitted
from exaflow.algorithms.federated.validation import check_interval


class MinMaxScaler(FeatureTransformer):
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        check_interval(feature_range)
        self.feature_range = tuple(float(value) for value in feature_range)

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
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
            feature_names=list(self.feature_names_in_),
            categorical_features=list(self.categorical_features_),
            primitive_name="minmax_scaler",
            location=self.data_min_.tolist(),
            scale=self._scale.tolist(),
            data_min=self.data_min_.tolist(),
            data_max=self.data_max_.tolist(),
            feature_range=self.feature_range,
        )
        return None

    @classmethod
    def from_recipe(cls, recipe):
        recipe = load_state(ScalerRecipe, recipe)
        if recipe.primitive_name != "minmax_scaler":
            raise BadInputError("Recipe does not belong to MinMaxScaler.")
        if recipe.data_min is None or recipe.data_max is None:
            raise BadInputError("MinMaxScaler recipe is missing data bounds.")
        scaler = cls(feature_range=recipe.feature_range or (0.0, 1.0))
        scaler.data_min_ = np.asarray(recipe.data_min, dtype=float)
        scaler.data_max_ = np.asarray(recipe.data_max, dtype=float)
        scaler._scale = np.asarray(recipe.scale, dtype=float)
        scaler._restore_feature_schema(recipe)
        scaler.recipe_ = recipe
        return scaler

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
        check_fitted(self, "data_min_", "data_max_", "_scale")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.data_min_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        result = (array - self.data_min_) * self._scale + self.feature_range[0]
        return _restore_shape(result, squeezed)
