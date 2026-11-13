"""SqrtTransformer: federated fit and named-column transformation."""

from __future__ import annotations

import numpy as np

from exaflow.algorithms.federated.feature_engineering.numerical import _as_2d_numeric
from exaflow.algorithms.federated.feature_engineering.numerical import _require_rows
from exaflow.algorithms.federated.feature_engineering.numerical import _restore_shape
from exaflow.algorithms.federated.feature_engineering.numerical import _validate_domain
from exaflow.algorithms.federated.feature_engineering.numerical import (
    _validate_local_domain,
)
from exaflow.algorithms.federated.feature_engineering.transformer import (
    FeatureTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import check_fitted


class SqrtTransformer(FeatureTransformer):
    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _validate_domain(
            array,
            agg_client,
            predicate=lambda x: x >= 0,
            message="sqrt requires non-negative values.",
        )
        self._shape = array.shape[1]
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
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
