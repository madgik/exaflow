"""Global standard scaling with learned parameters on the fitted object."""

import numpy as np

from exaflow.algorithms.federated.feature_engineering.numerical_utils import (
    as_2d_numeric,
)
from exaflow.algorithms.federated.feature_engineering.numerical_utils import (
    global_population_scale,
)
from exaflow.algorithms.federated.feature_engineering.transformer import (
    FeatureTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import as_numeric_array
from exaflow.algorithms.federated.validation import check_fitted


class FederatedStandardScaler(FeatureTransformer):
    def _validate_parameters(self, mean, scale, feature_count):
        mean = as_numeric_array(mean, input_name="Fitted mean")
        scale = as_numeric_array(scale, input_name="Fitted scale")
        if (
            mean.shape != (feature_count,)
            or scale.shape != (feature_count,)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise BadInputError(
                "Fitted parameters must match the features, with finite means "
                "and positive finite scales."
            )
        return mean, scale

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        selected = self._fit_data(data, categorical_vars, numerical_vars)
        array, _ = as_2d_numeric(selected)
        mean, scale = global_population_scale(array, agg_client)
        mean, scale = self._validate_parameters(mean, scale, array.shape[1])
        # Publish a complete fitted state only after all validation succeeds.
        self.mean_ = mean
        self.scale_ = scale
        self.feature_names_in_ = list(selected.columns)

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        selected = self._transform_data(data, categorical_vars, numerical_vars)
        check_fitted(self, "mean_", "scale_")
        array, _ = as_2d_numeric(selected)
        mean, scale = self._validate_parameters(self.mean_, self.scale_, array.shape[1])
        return (array - mean) / scale
