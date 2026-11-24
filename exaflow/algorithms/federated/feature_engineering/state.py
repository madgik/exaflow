"""Validation shared by the transformers' plain fitted-state dataclasses."""

from collections.abc import Mapping
from dataclasses import MISSING
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from math import isfinite
from numbers import Real

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import check_scalar


def load_state(state_type, value):
    if isinstance(value, state_type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise BadInputError("Provide fitted state as an object.")
    declared = {field.name for field in fields(state_type)}
    required = {
        field.name
        for field in fields(state_type)
        if field.default is MISSING and field.default_factory is MISSING
    }
    if set(value) - declared or required - set(value):
        raise BadInputError("Fitted state has missing or unexpected fields.")
    return state_type(**value)


def finite_scalar(value, *, name, **constraints):
    check_scalar(value, name=name, **constraints)
    try:
        representable = isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        representable = False
    if not representable:
        raise BadInputError(f"{name} must be finite in the numerical range.")


def finite_vector(values, *, name, minimum=None, length=None):
    if not isinstance(values, (list, tuple)) or not values:
        raise BadInputError(f"{name} must be a nonempty sequence.")
    if length is not None and len(values) != length:
        raise BadInputError(f"{name} must match the fitted feature count.")
    for value in values:
        if isinstance(value, bool):
            raise BadInputError(f"{name} must contain finite numbers.")
        finite_scalar(value, name=name, minimum=minimum)


def integer(value, *, name, minimum=0):
    # Protobuf Struct transports integer settings as exactly whole doubles.
    try:
        representable = isfinite(value)
    except (OverflowError, TypeError, ValueError):
        representable = False
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not representable
        or value != int(value)
        or value < minimum
    ):
        raise BadInputError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def ordered_pair(values, *, name):
    finite_vector(values, name=name, length=2)
    if values[0] > values[1]:
        raise BadInputError(f"{name} bounds must be ordered.")


@dataclass(kw_only=True)
class FittedState:
    feature_names: list[str]
    categorical_features: list[str]
    primitive_name: str
    schema_version: str = "2"

    def validate_header(self, names):
        if self.schema_version != "2" or self.primitive_name not in names:
            raise BadInputError("Unsupported fitted state kind or schema version.")
        if (
            not isinstance(self.feature_names, list)
            or not self.feature_names
            or any(
                not isinstance(name, str) or not name.strip()
                for name in self.feature_names
            )
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise BadInputError("Fitted feature names must be nonempty and unique.")
        if (
            not isinstance(self.categorical_features, list)
            or self.categorical_features
            != self.feature_names[: len(self.categorical_features)]
        ):
            raise BadInputError("Categorical features must match the fitted layout.")
