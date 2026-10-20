"""Reusable input checks independent of the execution runtime.

Algorithms retain their existing normalization before applying these checks.
Validation failures raise BadInputError locally. The caller owns cancellation
and cleanup of the distributed computation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from numbers import Real

import numpy as np
from sklearn.utils.validation import check_array as sklearn_check_array
from sklearn.utils.validation import check_consistent_length as sklearn_check_length
from sklearn.utils.validation import check_scalar as sklearn_check_scalar

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils import as_numeric_array


@dataclass(frozen=True)
class ArrayPolicy:
    dimensions: tuple[int, ...] = (2,)
    finite: bool = True


@dataclass(frozen=True)
class ParameterRule:
    types: tuple[type, ...]
    description: str


BOOLEAN_RULE = ParameterRule(
    types=(bool, np.bool_), description="a boolean (True or False)"
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    input_name: str
    message: str


class _ValidationError(BadInputError):
    def __init__(self, issue: ValidationIssue):
        super().__init__(issue.message)
        self.issue = issue


def _fail(code: str, input_name: str, message: str) -> None:
    raise _ValidationError(ValidationIssue(code, input_name, message)) from None


def validate_parameters(
    values: Mapping[str, object], rules: Mapping[str, ParameterRule]
) -> None:
    """Check current parameter types without coercing or changing parameters."""
    for name, rule in rules.items():
        if not isinstance(values[name], rule.types):
            _fail("invalid_parameter", name, f"{name} must be {rule.description}.")


def check_array(
    data, *, policy: ArrayPolicy, input_name: str, dtype=float
) -> np.ndarray:
    """Check numeric arrays without imposing sample or feature minima.

    Empty arrays and zero-feature matrices are deliberately allowed. Algorithms
    enforce their own local or global sample requirements separately.
    """
    dimensions = " or ".join(str(value) for value in policy.dimensions)
    message = f"{input_name} must be a numeric array with {dimensions} dimensions"
    if policy.finite:
        message += " containing only finite values"
    message += "."
    try:
        array = as_numeric_array(data, input_name=input_name, dtype=dtype)
    except BadInputError as exc:
        _fail("invalid_array", input_name, str(exc))
    try:
        result = sklearn_check_array(
            array,
            dtype=dtype,
            ensure_2d=False,
            allow_nd=True,
            ensure_all_finite=policy.finite,
            ensure_min_samples=0,
            ensure_min_features=0,
        )
    except (TypeError, ValueError):
        _fail("invalid_array", input_name, message)
    check_dimensions(result, policy.dimensions, input_name=input_name)
    return result


def check_dimensions(array, dimensions, *, input_name="Input", message=None) -> None:
    """Check rank independently of dtype, conversion, or missing-value policy."""
    if getattr(array, "ndim", None) not in dimensions:
        ranks = " or ".join(str(rank) for rank in dimensions)
        _fail(
            "invalid_array",
            input_name,
            message or f"{input_name} must be an array with {ranks} dimensions.",
        )


def check_consistent_length(*arrays, message=None) -> None:
    try:
        sklearn_check_length(*arrays)
    except ValueError:
        _fail(
            "inconsistent_length",
            "X_y",
            message
            or "Provide one outcome value for each observation in the feature matrix.",
        )


def check_binary_target(y: np.ndarray) -> None:
    """Check encoded values without requiring both classes locally."""
    if not np.isin(y, (0.0, 1.0)).all():
        _fail("invalid_binary_target", "y", "Outcome values must be encoded as 0 or 1.")


def check_classification_target(X, y) -> np.ndarray:
    """Retain label dtype without requiring local classes or dropping labels."""
    if y is None:
        raise BadInputError("Provide outcome labels for fitting.")
    try:
        labels = np.asarray(y)
    except (TypeError, ValueError):
        raise BadInputError(
            "Outcome must be a one-dimensional array of class labels."
        ) from None
    check_dimensions(labels, (1,), input_name="Outcome")
    check_consistent_length(X, labels)
    return labels


def check_feature_schema(X: np.ndarray, *, n_features: int) -> None:
    """A vector represents one observation; a matrix has features on axis 1."""
    actual = X.shape[-1]
    if actual != n_features:
        _fail(
            "inconsistent_features",
            "X",
            "Prediction features must match the fitted model. Provide the same "
            "features in the same order used for fitting.",
        )


def check_sample_counts(
    count: float, *, minimum: int, name: str, strict: bool = False
) -> None:
    """Apply a method's count requirement to an already computed count."""
    invalid = count <= minimum if strict else count < minimum
    if invalid:
        comparison = "more than" if strict else "at least"
        raise BadInputError(
            f"{name} requires {comparison} {minimum} observations. "
            "Select more observations or fewer covariates."
        )


def check_scalar(
    value,
    *,
    name: str,
    minimum=None,
    maximum=None,
    include_boundaries="both",
    target_type=Real,
    finite=True,
):
    """Validate an effective scalar parameter without changing or coercing it."""
    kind = "integer" if target_type is Integral else "number"
    conditions = []
    if minimum is not None:
        comparison = ">=" if include_boundaries in ("both", "left") else ">"
        conditions.append(f"{comparison} {minimum}")
    if maximum is not None:
        comparison = "<=" if include_boundaries in ("both", "right") else "<"
        conditions.append(f"{comparison} {maximum}")
    message = f"{name} must be a {'finite ' if finite else ''}{kind}"
    if conditions:
        message += " " + " and ".join(conditions)
    try:
        sklearn_check_scalar(
            value,
            name,
            target_type,
            min_val=minimum,
            max_val=maximum,
            include_boundaries=include_boundaries,
        )
        valid = isinstance(value, Integral) or bool(
            not np.isnan(value) and (not finite or np.isfinite(value))
        )
    except (TypeError, ValueError, OverflowError):
        _fail("invalid_parameter", name, message + ".")
    if not valid:
        _fail("invalid_parameter", name, message + ".")
    return value


def check_choice(value, choices, *, name: str) -> None:
    """Check a method's supported options without echoing supplied values."""
    if not isinstance(value, (str, int, float, np.generic)) or value not in choices:
        options = ", ".join(repr(choice) for choice in choices)
        _fail("invalid_parameter", name, f"{name} must be one of: {options}.")


def check_fitted(estimator, *attributes) -> None:
    """Check state on estimators that expose fitted attributes."""
    if any(not hasattr(estimator, attr) for attr in attributes):
        raise BadInputError(
            "This model is not fitted yet. Fit it before prediction or transformation."
        )


def check_dataframe(data, columns=()) -> None:
    """Validate frame access without changing missing-value or category policies."""
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise BadInputError("Provide the input data as a pandas DataFrame.")
    if any(column not in data.columns for column in columns):
        raise BadInputError("Input data is missing one or more required columns.")


def as_scalar(value, converter, *, name: str):
    """Retain an existing scalar conversion, reporting its expected failures safely."""
    try:
        return converter(value)
    except (TypeError, ValueError, OverflowError):
        _fail(
            "invalid_parameter",
            name,
            f"{name} must be a valid {converter.__name__} value.",
        )


def check_survival_target(y):
    """Parse the existing [time, event] target shared by the Cox estimators."""
    y = check_array(y, policy=ArrayPolicy(), input_name="Survival target")
    if y.shape[1] != 2:
        raise BadInputError(
            "Survival target y must have shape (n_obs, 2) with columns [time, event]."
        )
    if y.shape[0] == 0:
        raise BadInputError("Survival target y must contain at least one observation.")
    times = y[:, 0].reshape(-1)
    events = y[:, 1].reshape(-1)
    check_binary_target(events)
    return times, events


def check_survival_inputs(X, times, events, *, name):
    """Apply the Cox models' existing local data requirements."""
    X = check_array(X, policy=ArrayPolicy(), input_name="Features")
    times = as_numeric_array(times, input_name="Times").reshape(-1)
    events = as_numeric_array(events, input_name="Events").reshape(-1)
    check_array(times, policy=ArrayPolicy(dimensions=(1,)), input_name="Times")
    check_array(events, policy=ArrayPolicy(dimensions=(1,)), input_name="Events")
    check_consistent_length(X, times, events)
    if X.shape[0] == 0:
        raise BadInputError(f"{name} requires at least one observation.")
    if X.shape[1] == 0:
        raise BadInputError(f"{name} requires at least one covariate.")
    if np.min(times) <= 0:
        raise BadInputError("All time values must be strictly positive.")
    check_binary_target(events)
    return X, times, events


def check_feature_names(names, n_features):
    """Check explicit feature metadata without imposing uniqueness or ordering."""
    if names is not None and len(names) > 0 and len(names) != n_features:
        raise BadInputError("Provide one feature name for each feature column.")


def check_parameter_ranges(values, *, positive=(), nonnegative=()) -> None:
    """Check named numeric settings, including their finiteness, without casts."""
    for name in positive:
        check_scalar(values[name], name=name, minimum=0, include_boundaries="neither")
    for name in nonnegative:
        check_scalar(values[name], name=name, minimum=0)


def check_interval(bounds, *, name="bounds", strict=True, finite=True) -> None:
    """Check ordered limits; mathematical clipping bounds may be unbounded."""
    array = check_array(
        bounds, policy=ArrayPolicy(dimensions=(1,), finite=finite), input_name=name
    )
    if (
        array.size != 2
        or np.isnan(array).any()
        or (array[0] >= array[1] if strict else array[0] > array[1])
        or (array[0] == array[1] and not np.isfinite(array[0]))
    ):
        raise BadInputError(f"{name} must contain two ordered limits without NaN.")


def check_named_steps(steps, *, size=2) -> None:
    """Validate component names and tuple structure before fitting components."""
    names = []
    for step in steps:
        if not isinstance(step, (tuple, list)) or len(step) != size:
            raise BadInputError(
                "Provide each component as a named step with the required arguments."
            )
        name = step[0]
        if not isinstance(name, str) or not name:
            raise BadInputError("Each component must have a nonempty string name.")
        names.append(name)
    if len(set(names)) != len(names):
        raise BadInputError("Component names must be unique.")


def check_category_order(categories):
    """Return a validated pandas dtype without learning or sorting categories."""
    import pandas as pd

    if categories is None:
        raise BadInputError("Category ordering must be provided explicitly.")
    try:
        return pd.CategoricalDtype(categories=categories)
    except (TypeError, ValueError):
        raise BadInputError(
            "Category ordering must contain unique, non-missing category values."
        ) from None
