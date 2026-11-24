"""RobustScalerApprox: federated fit and named-column transformation."""

from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.feature_engineering.numerical import _as_2d_numeric
from exaflow.algorithms.federated.feature_engineering.numerical import _global_quartiles
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


class RobustScalerApprox(FeatureTransformer):
    @classmethod
    def from_recipe(cls, recipe):
        recipe = load_state(ScalerRecipe, recipe)
        if recipe.primitive_name != "robust_scaler":
            raise BadInputError("Recipe does not belong to RobustScalerApprox.")
        scaler = cls()
        scaler.center_ = np.asarray(recipe.location, dtype=float)
        scaler.scale_ = np.asarray(recipe.scale, dtype=float)
        scaler._restore_feature_schema(recipe)
        scaler.recipe_ = recipe
        return scaler

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        q25, self.center_, q75 = _global_quartiles(array, agg_client)
        self.scale_ = np.where(q75 - q25 == 0, 1.0, q75 - q25)
        self.recipe_ = ScalerRecipe(
            feature_names=list(self.feature_names_in_),
            categorical_features=list(self.categorical_features_),
            primitive_name="robust_scaler",
            location=self.center_.tolist(),
            scale=self.scale_.tolist(),
        )
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
        check_fitted(self, "center_", "scale_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.center_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        return _restore_shape((array - self.center_) / self.scale_, squeezed)
