"""OutlierDetector: federated fit and named-column transformation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exaflow.algorithms.federated.feature_engineering.numerical import _as_2d_numeric
from exaflow.algorithms.federated.feature_engineering.numerical import (
    _global_population_scale,
)
from exaflow.algorithms.federated.feature_engineering.numerical import _global_quartiles
from exaflow.algorithms.federated.feature_engineering.numerical import _reject_infinite
from exaflow.algorithms.federated.feature_engineering.numerical import _require_rows
from exaflow.algorithms.federated.feature_engineering.numerical import _restore_shape
from exaflow.algorithms.federated.feature_engineering.state import FittedState
from exaflow.algorithms.federated.feature_engineering.state import finite_scalar
from exaflow.algorithms.federated.feature_engineering.state import finite_vector
from exaflow.algorithms.federated.feature_engineering.state import load_state
from exaflow.algorithms.federated.feature_engineering.transformer import (
    FeatureTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import check_choice
from exaflow.algorithms.federated.validation import check_fitted


@dataclass(kw_only=True)
class OutlierDetectorRecipe(FittedState):
    mode: str
    parameters_by_feature: list[list[float]]
    threshold: float
    primitive_name: str = "outlier_detector"

    def __post_init__(self):
        self.validate_header(("outlier_detector",))
        check_choice(self.mode, ("zscore", "robust"), name="mode")
        finite_scalar(
            self.threshold, name="threshold", minimum=0, include_boundaries="neither"
        )
        if self.categorical_features:
            raise BadInputError("Outlier recipes require numerical features.")
        if not isinstance(self.parameters_by_feature, list) or len(
            self.parameters_by_feature
        ) != len(self.feature_names):
            raise BadInputError("Outlier parameters must match the feature count.")
        for row in self.parameters_by_feature:
            finite_vector(row, name="outlier parameters", length=2)
            if row[1] <= 0:
                raise BadInputError("Each outlier feature requires a positive scale.")


class OutlierDetector(FeatureTransformer):
    def __init__(self, mode: str = "zscore", threshold: float = 3.0):
        check_choice(mode, ("zscore", "robust"), name="mode")
        finite_scalar(
            threshold, name="threshold", minimum=0, include_boundaries="neither"
        )
        self.mode = mode
        self.threshold = float(threshold)

    @classmethod
    def from_recipe(cls, recipe):
        recipe = load_state(OutlierDetectorRecipe, recipe)
        detector = cls(mode=recipe.mode, threshold=recipe.threshold)
        parameters = np.asarray(recipe.parameters_by_feature, dtype=float)
        detector.location_ = parameters[:, 0]
        detector.scale_ = parameters[:, 1]
        detector._restore_feature_schema(recipe)
        detector.recipe_ = recipe
        return detector

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
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
            feature_names=list(self.feature_names_in_),
            categorical_features=list(self.categorical_features_),
            mode=self.mode,
            parameters_by_feature=parameters,
            threshold=self.threshold,
        )
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
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
