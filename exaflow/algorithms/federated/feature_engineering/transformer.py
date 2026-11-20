"""Named-column handling shared by numerical feature transformers."""

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer
from exaflow.algorithms.federated.validation import check_dataframe
from exaflow.algorithms.federated.validation import check_fitted


class FeatureTransformer(FederatedTransformer):
    """Select numerical columns without mutating the caller's dataframe."""

    def _feature_names(self, categorical_vars, numerical_vars):
        if not isinstance(categorical_vars, list) or not isinstance(
            numerical_vars, (list, type(None))
        ):
            raise BadInputError("Provide variable names as lists.")
        if categorical_vars:
            raise BadInputError("This transformer accepts numerical variables only.")
        names = list(numerical_vars or [])
        if (
            not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise BadInputError("Select nonempty, distinct feature names.")
        return names

    def _select_data(self, data, names):
        check_dataframe(data, names)
        if not data.columns.is_unique:
            raise BadInputError("Input dataframe column names must be unique.")
        return data.loc[:, names]

    def _fit_data(self, data, categorical_vars, numerical_vars):
        return self._select_data(
            data, self._feature_names(categorical_vars, numerical_vars)
        )

    def _transform_data(self, data, categorical_vars, numerical_vars):
        names = self.get_feature_names_out(
            categorical_vars=categorical_vars, numerical_vars=numerical_vars
        )
        return self._select_data(data, names)

    def get_feature_names_out(self, *, categorical_vars, numerical_vars=None):
        check_fitted(self, "feature_names_in_")
        names = self._feature_names(categorical_vars, numerical_vars)
        if names != self.feature_names_in_:
            raise BadInputError("Feature names and order must match the fitted state.")
        return names

    def fit_transform(self, *, agg_client, data, categorical_vars, numerical_vars=None):
        self.fit(
            agg_client=agg_client,
            data=data,
            categorical_vars=categorical_vars,
            numerical_vars=numerical_vars,
        )
        return self.transform(
            data, categorical_vars=categorical_vars, numerical_vars=numerical_vars
        )
