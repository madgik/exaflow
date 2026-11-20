"""MissingIndicator: federated fit and named-column transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.feature_engineering.numerical import _reject_infinite
from exaflow.algorithms.federated.feature_engineering.numerical import _require_rows
from exaflow.algorithms.federated.feature_engineering.numerical import _restore_shape
from exaflow.algorithms.federated.feature_engineering.transformer import (
    FeatureTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import check_fitted


class MissingIndicator(FeatureTransformer):
    _accepts_categorical = True

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
        if data.dtype.kind in "fiu":
            _reject_infinite(data)
        array = np.asarray(data, dtype=object)
        _require_rows(array)
        self._n_features = array.shape[1]
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
        if data.dtype.kind in "fiu":
            _reject_infinite(data)
        check_fitted(self, "_n_features")
        array = np.asarray(data, dtype=object)
        squeezed = False
        if getattr(self, "_n_features", None) != array.shape[1]:
            raise BadInputError(
                "Input feature count does not match the fitted indicator."
            )
        result = pd.isna(array).astype(int)
        return _restore_shape(result, squeezed)
