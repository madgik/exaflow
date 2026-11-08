"""JSON-serialisable fitted states for federated preprocessing primitives."""

from math import isfinite
from typing import Any
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from exaflow.algorithms.federated.preprocessing._categorical import category_key
from exaflow.algorithms.federated.preprocessing._categorical import validate_vocabulary


class FittedRecipe(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    schema_version: Literal["1"] = "1"
    primitive_name: str


class ScalerRecipe(FittedRecipe):
    primitive_name: Literal[
        "standard_scaler", "robust_scaler", "minmax_scaler", "maxabs_scaler"
    ]
    location: list[float] = Field(min_length=1)
    scale: list[float] = Field(min_length=1)
    data_min: Optional[list[float]] = None
    data_max: Optional[list[float]] = None
    feature_range: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def validate_shape(self):
        if len(self.location) != len(self.scale) or any(v <= 0 for v in self.scale):
            raise ValueError("Scaler locations and positive scales must align.")
        if self.primitive_name == "minmax_scaler":
            if (
                self.data_min is None
                or self.data_max is None
                or len(self.data_min) != len(self.scale)
                or len(self.data_max) != len(self.scale)
            ):
                raise ValueError("MinMax bounds must match the fitted feature count.")
            if any(low > high for low, high in zip(self.data_min, self.data_max)):
                raise ValueError("MinMax bounds must be ordered.")
            if (
                self.feature_range is None
                or self.feature_range[0] >= self.feature_range[1]
            ):
                raise ValueError("MinMax feature range must be ordered.")
        return self


class SimpleImputerRecipe(FittedRecipe):
    primitive_name: Literal["simple_imputer"] = "simple_imputer"
    strategy: Literal["mean", "median", "constant", "most_frequent"]
    statistics: list[Any] = Field(min_length=1)
    numeric_output: bool
    categories: Optional[list[list[Any]]] = None

    @model_validator(mode="after")
    def validate_statistics(self):
        if self.strategy in {"mean", "median"} and not self.numeric_output:
            raise ValueError("Mean and median recipes must produce numeric output.")
        if self.strategy == "most_frequent":
            categories = validate_vocabulary(self.categories)
            if self.numeric_output or len(categories) != len(self.statistics):
                raise ValueError(
                    "Categorical imputation state must match its vocabulary."
                )
            if any(
                category_key(value) not in {category_key(v) for v in column}
                for value, column in zip(self.statistics, categories)
            ):
                raise ValueError("Fitted modes must belong to the metadata vocabulary.")
        if any(value is None for value in self.statistics):
            raise ValueError("Imputation statistics must not be null.")
        if self.numeric_output:
            try:
                valid = all(isfinite(float(value)) for value in self.statistics)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError("Numeric imputation statistics must be finite.")
        return self


class WinsorizerRecipe(FittedRecipe):
    primitive_name: Literal["winsorizer"] = "winsorizer"
    lower_bounds: list[float] = Field(min_length=1)
    upper_bounds: list[float] = Field(min_length=1)
    lower_quantile: float = Field(ge=0, le=1)
    upper_quantile: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("Winsorizer quantiles must be ordered.")
        if len(self.lower_bounds) != len(self.upper_bounds) or any(
            low > high for low, high in zip(self.lower_bounds, self.upper_bounds)
        ):
            raise ValueError("Winsorizer bounds must align and be ordered.")
        return self


class OutlierDetectorRecipe(FittedRecipe):
    primitive_name: Literal["outlier_detector"] = "outlier_detector"
    mode: Literal["zscore", "robust"]
    parameters_by_feature: list[list[float]] = Field(min_length=1)
    threshold: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_parameters(self):
        if any(len(row) != 2 or row[1] <= 0 for row in self.parameters_by_feature):
            raise ValueError(
                "Each outlier feature requires a location and positive scale."
            )
        return self


class YeoJohnsonFitDiagnostic(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    status: Literal["converged", "approximate", "constant"]
    reason: Literal[
        "convergence",
        "flat_objective",
        "evaluation_budget_exhausted",
        "numerical_bound",
        "constant_feature",
    ]
    evaluations: int = Field(ge=0)
    searched_bounds: tuple[float, float]
    safe_bounds: tuple[float, float]
    objective: Optional[float] = None

    @model_validator(mode="after")
    def validate_termination(self):
        if (
            self.safe_bounds[0] > self.safe_bounds[1]
            or self.searched_bounds[0] > self.searched_bounds[1]
        ):
            raise ValueError("Search diagnostic bounds must be ordered.")
        if self.status == "constant":
            if (
                self.reason != "constant_feature"
                or self.evaluations != 0
                or self.objective is not None
            ):
                raise ValueError("Constant features do not require search evaluations.")
        elif self.evaluations < 1 or self.objective is None:
            raise ValueError(
                "A fitted search requires finite objective and evaluations."
            )
        if self.status == "converged" and self.reason != "convergence":
            raise ValueError("Converged fits must report convergence.")
        if self.status == "approximate" and self.reason not in {
            "flat_objective",
            "evaluation_budget_exhausted",
            "numerical_bound",
        }:
            raise ValueError("Approximate fits must report their limitation.")
        return self


class YeoJohnsonRecipe(FittedRecipe):
    primitive_name: Literal["yeo_johnson"] = "yeo_johnson"
    lambdas: list[float] = Field(min_length=1)
    max_evaluations: int = Field(ge=1)
    lambda_tolerance: float = Field(gt=0)
    likelihood_tolerance: float = Field(ge=0)
    fit_diagnostics: list[YeoJohnsonFitDiagnostic] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_search_state(self):
        if len(self.fit_diagnostics) != len(self.lambdas):
            raise ValueError("Provide one search diagnostic per fitted feature.")
        for lambd, diagnostic in zip(self.lambdas, self.fit_diagnostics):
            if diagnostic.status == "constant" and lambd != 1:
                raise ValueError("Constant features must use the identity lambda.")
            if (
                diagnostic.searched_bounds[0] < diagnostic.safe_bounds[0]
                or diagnostic.searched_bounds[1] > diagnostic.safe_bounds[1]
            ):
                raise ValueError("Recorded search must stay inside numeric bounds.")
            if diagnostic.evaluations > self.max_evaluations:
                raise ValueError("Fitted search must respect its evaluation budget.")
            if not (
                diagnostic.safe_bounds[0] <= lambd <= diagnostic.safe_bounds[1]
                and diagnostic.searched_bounds[0]
                <= lambd
                <= diagnostic.searched_bounds[1]
            ):
                raise ValueError(
                    "Fitted lambda must belong to the recorded search bounds."
                )
        return self


class QuantileTransformerRecipe(FittedRecipe):
    primitive_name: Literal["quantile_transformer_approx"] = (
        "quantile_transformer_approx"
    )
    quantiles: list[list[float]] = Field(min_length=1)
    references: list[float] = Field(min_length=2)
    output_distribution: Literal["uniform", "normal"]
    min_observations_per_interval: int = Field(ge=1)
    platform_minimum: int = Field(ge=1)
    n_bins: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_grid(self):
        if (
            self.references[0] != 0
            or self.references[-1] != 1
            or any(a >= b for a, b in zip(self.references, self.references[1:]))
        ):
            raise ValueError("Quantile references must increase from zero to one.")
        if any(
            len(row) != len(self.references) or any(a > b for a, b in zip(row, row[1:]))
            for row in self.quantiles
        ):
            raise ValueError("Quantile rows must be ordered and match the references.")
        if self.min_observations_per_interval < self.platform_minimum:
            raise ValueError("Quantile recipe must respect the platform minimum.")
        return self
