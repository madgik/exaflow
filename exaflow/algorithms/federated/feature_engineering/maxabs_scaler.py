"""MaxAbsScaler: federated fit and named-column transformation."""

from __future__ import annotations

from math import isfinite

import numpy as np

from exaflow.algorithms.federated.feature_engineering.numerical import _as_2d_numeric
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


class MaxAbsScaler(FeatureTransformer):
    @classmethod
    def from_recipe(cls, recipe):
        recipe = load_state(ScalerRecipe, recipe)
        if recipe.primitive_name != "maxabs_scaler":
            raise BadInputError("Recipe does not belong to MaxAbsScaler.")
        scaler = cls()
        scaler.scale_ = np.asarray(recipe.scale, dtype=float)
        scaler.max_abs_ = scaler.scale_.copy()
        scaler._restore_feature_schema(recipe)
        scaler.recipe_ = recipe
        return scaler

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
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
            feature_names=list(self.feature_names_in_),
            categorical_features=list(self.categorical_features_),
            primitive_name="maxabs_scaler",
            location=[0.0] * len(self.scale_),
            scale=self.scale_.tolist(),
        )
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
        check_fitted(self, "scale_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.scale_):
            raise BadInputError("Input feature count does not match the fitted scaler.")
        return _restore_shape(array / self.scale_, squeezed)
