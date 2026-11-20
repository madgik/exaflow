"""SimpleImputer: federated fit and named-column transformation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Number
from typing import Any
from typing import Optional

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.feature_engineering.numerical import _as_2d_numeric
from exaflow.algorithms.federated.feature_engineering.numerical import _global_mean
from exaflow.algorithms.federated.feature_engineering.numerical import _percentile
from exaflow.algorithms.federated.feature_engineering.numerical import _reject_infinite
from exaflow.algorithms.federated.feature_engineering.numerical import _require_rows
from exaflow.algorithms.federated.feature_engineering.numerical import (
    check_common_configuration,
)
from exaflow.algorithms.federated.feature_engineering.state import FittedState
from exaflow.algorithms.federated.feature_engineering.state import finite_vector
from exaflow.algorithms.federated.feature_engineering.state import load_state
from exaflow.algorithms.federated.feature_engineering.transformer import (
    FeatureTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import check_choice
from exaflow.algorithms.federated.validation import check_fitted
from exaflow.algorithms.federated.validation import check_scalar


@dataclass(kw_only=True)
class SimpleImputerRecipe(FittedState):
    strategy: str
    statistics: list
    numeric_output: bool
    categories: list[list] | None = None
    primitive_name: str = "simple_imputer"

    def __post_init__(self):
        self.validate_header(("simple_imputer",))
        check_choice(
            self.strategy,
            ("mean", "median", "constant", "most_frequent"),
            name="strategy",
        )
        if type(self.numeric_output) is not bool:
            raise BadInputError("numeric_output must be boolean.")
        if not isinstance(self.statistics, list) or len(self.statistics) != len(
            self.feature_names
        ):
            raise BadInputError("Imputation statistics must match the feature count.")
        if self.strategy in {"mean", "median"} and (
            not self.numeric_output or self.categorical_features
        ):
            raise BadInputError("Mean and median recipes require numerical features.")
        if self.strategy == "most_frequent":
            categories = validate_vocabulary(self.categories)
            if self.categorical_features != self.feature_names:
                raise BadInputError(
                    "Categorical recipes require categorical feature declarations."
                )
            if self.numeric_output or len(categories) != len(self.statistics):
                raise BadInputError(
                    "Categorical imputation state must match its vocabulary."
                )
            if any(
                category_key(value) not in {category_key(v) for v in column}
                for value, column in zip(self.statistics, categories)
            ):
                raise BadInputError(
                    "Fitted modes must belong to the metadata vocabulary."
                )
        elif self.categories is not None:
            validate_vocabulary(self.categories)
        for value in self.statistics:
            category_key(value)
        if self.numeric_output:
            finite_vector(self.statistics, name="statistics")


def category_key(value):
    """Keep numeric codes, text codes and booleans distinct."""
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (str, int, float, bool):
        raise BadInputError("Categories must be finite JSON scalar values.")
    if isinstance(value, float) and not isfinite(value):
        raise BadInputError("Categories must be finite JSON scalar values.")
    return ("number" if type(value) in (int, float) else type(value).__name__), value


def validate_vocabulary(categories):
    if not isinstance(categories, (list, tuple)) or not categories:
        raise BadInputError("Categorical most_frequent requires a metadata vocabulary.")
    validated = []
    for column in categories:
        if not isinstance(column, (list, tuple)) or not column:
            raise BadInputError("Provide a non-empty metadata vocabulary per feature.")
        keys = [category_key(value) for value in column]
        if len(set(keys)) != len(keys):
            raise BadInputError("Metadata vocabulary entries must be unique.")
        validated.append([key[1] for key in keys])
    return validated


def vocabulary_counts(values, categories):
    """Validate every observed value before any frequency counts are shared."""
    if values.ndim != 2 or values.shape[1] != len(categories):
        raise BadInputError("Metadata vocabulary must match the feature count.")
    counts = []
    for idx, column in enumerate(categories):
        positions = {category_key(value): i for i, value in enumerate(column)}
        local = np.zeros(len(column), dtype=float)
        for value in values[:, idx]:
            if value is None or value is pd.NA:
                continue
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                continue
            key = category_key(value)
            if key not in positions:
                raise BadInputError(
                    "Observed values are outside the metadata vocabulary."
                )
            local[positions[key]] += 1
        counts.append(local)
    return counts


class SimpleImputer(FeatureTransformer):
    """Impute missing values using globally fitted statistics.

    Constant imputation defaults to ``0`` for numeric input when no fill value
    is supplied. Categorical mode requires one ordered vocabulary per feature,
    supplied by a trusted metadata-aware caller. Unknown values are rejected;
    the primitive never discovers categories from observed data.
    """

    _accepts_categorical = True

    def __init__(
        self, strategy: str = "mean", fill_value: Any = None, *, categories=None
    ):
        check_choice(
            strategy, ("mean", "median", "constant", "most_frequent"), name="strategy"
        )
        self.strategy = strategy
        self.fill_value = fill_value
        self.categories = categories
        self.statistics_: Optional[np.ndarray] = None

    @classmethod
    def from_recipe(cls, recipe):
        recipe = load_state(SimpleImputerRecipe, recipe)
        imputer = cls(strategy=recipe.strategy, categories=recipe.categories)
        imputer.statistics_ = np.asarray(recipe.statistics, dtype=object)
        imputer._numeric_output = recipe.numeric_output
        imputer.categories_ = recipe.categories
        imputer._restore_feature_schema(recipe)
        imputer.recipe_ = recipe
        return imputer

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
        if self.strategy in {"mean", "median"} and categorical_vars:
            raise BadInputError("Mean and median require numerical variables.")
        if self.strategy == "constant":
            values = np.asarray(data, dtype=object)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.ndim != 2:
                raise BadInputError("Input must be one- or two-dimensional.")
            _require_rows(values)
            if values.shape[1] == 0:
                raise BadInputError("Input must contain at least one feature.")
            observed = values[~pd.isna(values)]
            self._numeric_output = not categorical_vars and all(
                isinstance(value, Number) for value in observed
            )
            fill_value = self.fill_value
            if fill_value is None:
                if not self._numeric_output:
                    raise BadInputError(
                        "fill_value is required for non-numeric constant imputation."
                    )
                fill_value = 0
            if self._numeric_output:
                numeric, _ = _as_2d_numeric(data)
                _reject_infinite(numeric)
                check_scalar(fill_value, name="fill_value")
            self.statistics_ = np.asarray([fill_value] * values.shape[1], dtype=object)
        elif self.strategy in {"mean", "median"}:
            self._numeric_output = True
            array, _ = _as_2d_numeric(data)
            _require_rows(array)
            _reject_infinite(array)
            if self.strategy == "mean":
                self.statistics_ = _global_mean(array, agg_client)
            else:
                self.statistics_ = np.asarray(
                    [
                        _percentile(array[:, idx], 0.5, agg_client)
                        for idx in range(array.shape[1])
                    ]
                )
        else:
            self._numeric_output = False
            # Reject schema/value errors collectively before anyone sends counts.
            errors = np.zeros(2, dtype=float)
            try:
                if list(categorical_vars) != self.feature_names_in_:
                    raise BadInputError(
                        "Categorical mode requires categorical feature declarations."
                    )
                categories = validate_vocabulary(self.categories)
                values = np.asarray(data, dtype=object)
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                if values.ndim != 2 or values.shape[1] != len(categories):
                    raise BadInputError("Vocabulary must match the feature count.")
            except (BadInputError, TypeError, ValueError):
                errors[0] = 1
            if not errors[0]:
                try:
                    local_counts = vocabulary_counts(values, categories)
                except BadInputError:
                    errors[1] = 1
            global_errors = np.asarray(agg_client.sum(errors), dtype=float)
            if global_errors[0]:
                raise BadInputError(
                    "Categorical most_frequent requires a complete metadata vocabulary."
                )
            if global_errors[1]:
                raise BadInputError(
                    "Observed values are outside the metadata vocabulary."
                )
            check_common_configuration(categories, agg_client)
            statistics = []
            for column, local in zip(categories, local_counts):
                counts = np.asarray(agg_client.sum(local), dtype=float)
                if counts.sum() <= 0:
                    raise BadInputError(
                        "At least one feature has no valid observations."
                    )
                statistics.append(column[int(np.argmax(counts))])
            self.categories_ = categories
            self.statistics_ = np.asarray(statistics, dtype=object)
        self.recipe_ = SimpleImputerRecipe(
            feature_names=list(self.feature_names_in_),
            categorical_features=list(self.categorical_features_),
            strategy=self.strategy,
            statistics=self.statistics_.tolist(),
            numeric_output=self._numeric_output,
            categories=self.categories_ if self.strategy == "most_frequent" else None,
        )
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
        check_fitted(self, "statistics_")
        if self.statistics_ is None:
            raise BadInputError("SimpleImputer is not fitted.")
        values = np.asarray(data, dtype=object).copy()
        squeezed = values.ndim == 1
        if squeezed:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != len(self.statistics_):
            raise BadInputError(
                "Input feature count does not match the fitted imputer."
            )
        if self.strategy == "most_frequent":
            vocabulary_counts(values, self.categories_)
        for idx, statistic in enumerate(self.statistics_):
            missing = pd.isna(values[:, idx])
            values[missing, idx] = statistic
        if self._numeric_output:
            values, _ = _as_2d_numeric(values)
            _reject_infinite(values)
        return values[:, 0] if squeezed else values
