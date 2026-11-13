"""Winsorizer: federated fit and named-column transformation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exaflow.algorithms.federated.feature_engineering.numerical import _as_2d_numeric
from exaflow.algorithms.federated.feature_engineering.numerical import _percentile
from exaflow.algorithms.federated.feature_engineering.numerical import _reject_infinite
from exaflow.algorithms.federated.feature_engineering.numerical import _require_rows
from exaflow.algorithms.federated.feature_engineering.numerical import _restore_shape
from exaflow.algorithms.federated.feature_engineering.state import FittedState
from exaflow.algorithms.federated.feature_engineering.state import finite_vector
from exaflow.algorithms.federated.feature_engineering.state import load_state
from exaflow.algorithms.federated.feature_engineering.transformer import (
    FeatureTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import check_fitted
from exaflow.algorithms.federated.validation import check_scalar


@dataclass(kw_only=True)
class WinsorizerRecipe(FittedState):
    lower_bounds: list[float]
    upper_bounds: list[float]
    lower_quantile: float
    upper_quantile: float
    primitive_name: str = "winsorizer"

    def __post_init__(self):
        self.validate_header(("winsorizer",))
        if self.categorical_features:
            raise BadInputError("Winsorizer recipes require numerical features.")
        finite_vector(
            self.lower_bounds, name="lower bounds", length=len(self.feature_names)
        )
        finite_vector(
            self.upper_bounds, name="upper bounds", length=len(self.feature_names)
        )
        check_scalar(self.lower_quantile, name="lower_quantile", minimum=0, maximum=1)
        check_scalar(self.upper_quantile, name="upper_quantile", minimum=0, maximum=1)
        if self.lower_quantile >= self.upper_quantile:
            raise BadInputError("Winsorizer quantiles must be ordered.")
        if any(low > high for low, high in zip(self.lower_bounds, self.upper_bounds)):
            raise BadInputError("Winsorizer bounds must be ordered.")


class Winsorizer(FeatureTransformer):
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
        recipe = load_state(WinsorizerRecipe, recipe)
        winsorizer = cls(recipe.lower_quantile, recipe.upper_quantile)
        winsorizer.lower_bounds_ = np.asarray(recipe.lower_bounds, dtype=float)
        winsorizer.upper_bounds_ = np.asarray(recipe.upper_bounds, dtype=float)
        winsorizer._restore_feature_schema(recipe)
        winsorizer.recipe_ = recipe
        return winsorizer

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
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
            feature_names=list(self.feature_names_in_),
            categorical_features=list(self.categorical_features_),
            lower_bounds=self.lower_bounds_.tolist(),
            upper_bounds=self.upper_bounds_.tolist(),
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile,
        )
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
        check_fitted(self, "lower_bounds_", "upper_bounds_")
        array, squeezed = _as_2d_numeric(data)
        if array.shape[1] != len(self.lower_bounds_):
            raise BadInputError(
                "Input feature count does not match the fitted winsorizer."
            )
        return _restore_shape(
            np.clip(array, self.lower_bounds_, self.upper_bounds_), squeezed
        )
