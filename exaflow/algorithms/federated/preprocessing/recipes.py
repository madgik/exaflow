"""JSON-serialisable fitted states for federated preprocessing primitives."""

from typing import Any
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class FittedRecipe(BaseModel):
    schema_version: str = "1"
    primitive_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ScalerRecipe(FittedRecipe):
    primitive_name: Literal[
        "standard_scaler", "robust_scaler", "minmax_scaler", "maxabs_scaler"
    ]
    location: list[float] = Field(default_factory=list)
    scale: list[float] = Field(default_factory=list)
    data_min: Optional[list[float]] = None
    data_max: Optional[list[float]] = None
    feature_range: Optional[tuple[float, float]] = None


class SimpleImputerRecipe(FittedRecipe):
    primitive_name: Literal["simple_imputer"] = "simple_imputer"
    strategy: str
    statistics: list[Any]


class WinsorizerRecipe(FittedRecipe):
    primitive_name: Literal["winsorizer"] = "winsorizer"
    lower_bounds: list[float]
    upper_bounds: list[float]
    lower_quantile: float
    upper_quantile: float


class OutlierDetectorRecipe(FittedRecipe):
    primitive_name: Literal["outlier_detector"] = "outlier_detector"
    mode: str
    parameters_by_feature: list[list[float]]
    threshold: float


class YeoJohnsonRecipe(FittedRecipe):
    primitive_name: Literal["yeo_johnson"] = "yeo_johnson"
    lambdas: list[float]


class QuantileTransformerRecipe(FittedRecipe):
    primitive_name: Literal["quantile_transformer_approx"] = (
        "quantile_transformer_approx"
    )
    quantiles: list[list[float]]
    references: list[float]
    output_distribution: str
    min_observations_per_interval: int = 5
