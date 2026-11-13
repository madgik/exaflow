"""Fitted state shared by the four scaling methods."""

from dataclasses import dataclass

from exaflow.algorithms.federated.utils import BadInputError

from .state import FittedState
from .state import finite_vector


@dataclass(kw_only=True)
class ScalerRecipe(FittedState):
    location: list[float]
    scale: list[float]
    data_min: list[float] | None = None
    data_max: list[float] | None = None
    feature_range: tuple[float, float] | None = None

    def __post_init__(self):
        self.validate_header(
            ("standard_scaler", "robust_scaler", "minmax_scaler", "maxabs_scaler")
        )
        n = len(self.feature_names)
        if self.categorical_features:
            raise BadInputError("Scaler recipes require numerical features.")
        finite_vector(self.location, name="location", length=n)
        finite_vector(self.scale, name="scale", length=n)
        if any(v <= 0 for v in self.scale):
            raise BadInputError("Scaler scales must be positive.")
        for bounds in (self.data_min, self.data_max):
            if bounds is not None:
                finite_vector(bounds, name="data bounds", length=n)
        if self.feature_range is not None:
            finite_vector(self.feature_range, name="feature_range", length=2)
            self.feature_range = tuple(self.feature_range)
            if self.feature_range[0] >= self.feature_range[1]:
                raise BadInputError("MinMax feature range must be ordered.")
        if self.primitive_name == "minmax_scaler":
            if (
                self.data_min is None
                or self.data_max is None
                or self.feature_range is None
            ):
                raise BadInputError("MinMax recipe requires bounds and feature range.")
            if any(low > high for low, high in zip(self.data_min, self.data_max)):
                raise BadInputError("MinMax bounds must be ordered.")
