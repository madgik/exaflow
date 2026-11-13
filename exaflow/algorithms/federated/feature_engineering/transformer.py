"""Shared named-column contract for the numerical feature transformers."""

import numpy as np

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer
from exaflow.algorithms.federated.validation import check_dataframe
from exaflow.algorithms.federated.validation import check_fitted


class FeatureTransformer(FederatedTransformer):
    """Implement column selection, not an adapter around another transformer.

    Subclasses implement fit/transform themselves. Their numerical calculations
    operate only on the selected columns and never mutate the caller's frame.
    """

    _accepts_categorical = False

    def _feature_names(self, categorical_vars, numerical_vars):
        if not isinstance(categorical_vars, list) or (
            numerical_vars is not None and not isinstance(numerical_vars, list)
        ):
            raise BadInputError(
                "Provide categorical and numerical variable names as lists."
            )
        categorical = list(categorical_vars)
        numerical = list(numerical_vars or [])
        if categorical and not self._accepts_categorical:
            raise BadInputError("This transformer accepts numerical variables only.")
        names = categorical + numerical
        if (
            not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise BadInputError("Select nonempty, distinct feature names.")
        return names

    def _fit_data(self, data, categorical_vars, numerical_vars):
        names = self._feature_names(categorical_vars, numerical_vars)
        check_dataframe(data, names)
        if not data.columns.is_unique:
            raise BadInputError("Input dataframe column names must be unique.")
        self.feature_names_in_ = names
        self.categorical_features_ = list(categorical_vars)
        return data.loc[:, names].to_numpy()

    def _transform_data(self, data, categorical_vars, numerical_vars):
        names = self.get_feature_names_out(
            categorical_vars=categorical_vars, numerical_vars=numerical_vars
        )
        check_dataframe(data, names)
        if not data.columns.is_unique:
            raise BadInputError("Input dataframe column names must be unique.")
        return data.loc[:, names].to_numpy()

    def _restore_feature_schema(self, recipe):
        self.feature_names_in_ = list(recipe.feature_names)
        self.categorical_features_ = list(recipe.categorical_features)

    def get_feature_names_out(self, *, categorical_vars, numerical_vars=None):
        check_fitted(self, "feature_names_in_", "categorical_features_")
        names = self._feature_names(categorical_vars, numerical_vars)
        if (
            names != self.feature_names_in_
            or list(categorical_vars) != self.categorical_features_
        ):
            raise BadInputError("Feature names and order must match the fitted state.")
        return list(names)

    def fit_transform(
        self, *, agg_client, data, categorical_vars, numerical_vars=None
    ) -> np.ndarray:
        self.fit(
            agg_client=agg_client,
            data=data,
            categorical_vars=categorical_vars,
            numerical_vars=numerical_vars,
        )
        return self.transform(
            data, categorical_vars=categorical_vars, numerical_vars=numerical_vars
        )
