"""StandardScaler: federated fit and named-column transformation."""

from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.feature_engineering.numerical import _as_2d_numeric
from exaflow.algorithms.federated.feature_engineering.numerical import (
    _global_population_scale,
)
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


class StandardScaler(FeatureTransformer):
    @classmethod
    def from_recipe(cls, recipe):
        recipe = load_state(ScalerRecipe, recipe)
        if recipe.primitive_name != "standard_scaler":
            raise BadInputError("Recipe does not belong to StandardScaler.")
        scaler = cls()
        scaler.mean_ = np.asarray(recipe.location, dtype=float)
        scaler.scale_ = np.asarray(recipe.scale, dtype=float)
        scaler._restore_feature_schema(recipe)
        scaler.recipe_ = recipe
        return scaler

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        self.mean_, self.scale_ = _global_population_scale(array, agg_client)
        self.recipe_ = ScalerRecipe(
            feature_names=list(self.feature_names_in_),
            categorical_features=list(self.categorical_features_),
            primitive_name="standard_scaler",
            location=self.mean_.tolist(),
            scale=self.scale_.tolist(),
        )
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
        check_fitted(self, "mean_", "scale_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.mean_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        return _restore_shape((array - self.mean_) / self.scale_, squeezed)
