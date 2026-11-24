"""QuantileTransformerApprox: federated fit and named-column transformation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.special import ndtri

from exaflow.algorithms.federated.feature_engineering.numerical import _as_2d_numeric
from exaflow.algorithms.federated.feature_engineering.numerical import _global_min_max
from exaflow.algorithms.federated.feature_engineering.numerical import _reject_infinite
from exaflow.algorithms.federated.feature_engineering.numerical import _require_rows
from exaflow.algorithms.federated.feature_engineering.numerical import _restore_shape
from exaflow.algorithms.federated.feature_engineering.numerical import (
    check_common_configuration,
)
from exaflow.algorithms.federated.feature_engineering.state import FittedState
from exaflow.algorithms.federated.feature_engineering.state import finite_scalar
from exaflow.algorithms.federated.feature_engineering.state import finite_vector
from exaflow.algorithms.federated.feature_engineering.state import integer
from exaflow.algorithms.federated.feature_engineering.state import load_state
from exaflow.algorithms.federated.feature_engineering.state import ordered_pair
from exaflow.algorithms.federated.feature_engineering.transformer import (
    FeatureTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.validation import check_choice
from exaflow.algorithms.federated.validation import check_fitted

from .batched_quantiles import batched_quantiles


@dataclass(kw_only=True)
class QuantileFitDiagnostic:
    histogram_rounds: int
    intervals: list[tuple[float, float]]
    support_counts: list[int]
    max_interval_width: float
    limited_by_minimum_support: bool
    limited_by_refinement_budget: bool

    def __post_init__(self):
        self.histogram_rounds = integer(self.histogram_rounds, name="histogram_rounds")
        if not isinstance(self.intervals, list) or len(self.intervals) < 2:
            raise BadInputError("Quantile intervals must contain endpoints.")
        if not isinstance(self.support_counts, list) or len(self.intervals) != len(
            self.support_counts
        ):
            raise BadInputError("Quantile support counts must match the intervals.")
        for pair in self.intervals:
            ordered_pair(pair, name="quantile interval")
        self.support_counts = [
            integer(count, name="support_count") for count in self.support_counts
        ]
        if self.support_counts[0] != 0 or self.support_counts[-1] != 0:
            raise BadInputError(
                "Global endpoints do not have per-group support counts."
            )
        finite_scalar(self.max_interval_width, name="max_interval_width", minimum=0)
        if self.max_interval_width != max(high - low for low, high in self.intervals):
            raise BadInputError("Quantile uncertainty width must match the intervals.")
        if (
            type(self.limited_by_minimum_support) is not bool
            or type(self.limited_by_refinement_budget) is not bool
        ):
            raise BadInputError("Quantile limitation flags must be boolean.")


@dataclass(kw_only=True)
class QuantileTransformerRecipe(FittedState):
    quantiles: list[list[float]]
    references: list[float]
    output_distribution: str
    min_observations_per_interval: int
    platform_minimum: int
    n_bins: int
    max_refinement_rounds: int
    fit_diagnostics: list[QuantileFitDiagnostic]
    primitive_name: str = "quantile_transformer_approx"

    def __post_init__(self):
        self.validate_header(("quantile_transformer_approx",))
        if self.categorical_features:
            raise BadInputError("Quantile recipes require numerical features.")
        finite_vector(self.references, name="references")
        if (
            len(self.references) < 2
            or self.references[0] != 0
            or self.references[-1] != 1
            or any(a >= b for a, b in zip(self.references, self.references[1:]))
        ):
            raise BadInputError("Quantile references must increase from zero to one.")
        check_choice(
            self.output_distribution, ("uniform", "normal"), name="output_distribution"
        )
        for name, minimum in (
            ("min_observations_per_interval", 1),
            ("platform_minimum", 1),
            ("n_bins", 2),
            ("max_refinement_rounds", 0),
        ):
            setattr(
                self, name, integer(getattr(self, name), name=name, minimum=minimum)
            )
        if self.min_observations_per_interval < self.platform_minimum:
            raise BadInputError("Quantile recipe must respect the platform minimum.")
        if not isinstance(self.quantiles, list) or len(self.quantiles) != len(
            self.feature_names
        ):
            raise BadInputError("Quantiles must match the feature count.")
        if not isinstance(self.fit_diagnostics, list) or len(
            self.fit_diagnostics
        ) != len(self.quantiles):
            raise BadInputError("Provide one quantile diagnostic per feature.")
        self.fit_diagnostics = [
            load_state(QuantileFitDiagnostic, d) for d in self.fit_diagnostics
        ]
        for row, diagnostic in zip(self.quantiles, self.fit_diagnostics):
            finite_vector(row, name="quantiles", length=len(self.references))
            if any(a > b for a, b in zip(row, row[1:])):
                raise BadInputError("Quantile rows must be ordered.")
            if len(diagnostic.intervals) != len(self.references):
                raise BadInputError("Quantile intervals must match the references.")
            if diagnostic.histogram_rounds > self.max_refinement_rounds + 1:
                raise BadInputError(
                    "Quantile diagnostics exceed the refinement budget."
                )
            if any(
                count < self.min_observations_per_interval
                for count in diagnostic.support_counts[1:-1]
            ):
                raise BadInputError(
                    "Interior quantile groups must respect minimum support."
                )
            if any(
                not low <= value <= high
                for value, (low, high) in zip(row, diagnostic.intervals)
            ):
                raise BadInputError(
                    "Quantile estimates must belong to their intervals."
                )


class QuantileTransformerApprox(FeatureTransformer):
    """Map values through approximate global quantiles.

    The trusted caller supplies the platform privacy minimum. Fitting requires
    at least max(platform_minimum, min_observations_per_interval) observations
    per requested rank interval. This is a cohort-size guard, not differential
    privacy or a guarantee on the occupancy of value bins.

    Quantiles share batched histogram refinement, with two additional rounds by
    default. Selected bins are enlarged to meet the effective support floor;
    only supported groups expose aggregate extrema. Recipes retain achieved
    uncertainty intervals instead of claiming a fixed accuracy. Exact global
    endpoints are retained separately from the interior support rule.
    """

    def __init__(
        self,
        n_quantiles: int = 100,
        output_distribution: str = "uniform",
        min_observations_per_interval: int = 5,
        *,
        platform_minimum: int,
        n_bins: int = 256,
        max_refinement_rounds: int = 2,
    ):
        for name, value, minimum in (
            ("n_quantiles", n_quantiles, 2),
            ("min_observations_per_interval", min_observations_per_interval, 1),
            ("platform_minimum", platform_minimum, 1),
            ("n_bins", n_bins, 2),
            ("max_refinement_rounds", max_refinement_rounds, 0),
        ):
            if isinstance(value, (bool, np.bool_)):
                raise BadInputError(f"{name} must be an integer.")
            finite_scalar(value, name=name, target_type=Integral, minimum=minimum)
        check_choice(
            output_distribution, ("uniform", "normal"), name="output_distribution"
        )
        self.n_quantiles = int(n_quantiles)
        self.output_distribution = output_distribution
        self.min_observations_per_interval = int(min_observations_per_interval)
        self.platform_minimum = int(platform_minimum)
        self.n_bins = int(n_bins)
        self.max_refinement_rounds = int(max_refinement_rounds)

    @classmethod
    def from_recipe(cls, recipe):
        recipe = load_state(QuantileTransformerRecipe, recipe)
        transformer = cls(
            n_quantiles=len(recipe.references),
            output_distribution=recipe.output_distribution,
            min_observations_per_interval=recipe.min_observations_per_interval,
            platform_minimum=recipe.platform_minimum,
            n_bins=recipe.n_bins,
            max_refinement_rounds=recipe.max_refinement_rounds,
        )
        transformer.quantiles_ = np.asarray(recipe.quantiles, dtype=float)
        transformer.references_ = np.asarray(recipe.references, dtype=float)
        transformer.fit_diagnostics_ = recipe.fit_diagnostics
        transformer._restore_feature_schema(recipe)
        transformer.recipe_ = recipe
        return transformer

    def fit(self, *, agg_client, data, categorical_vars, numerical_vars=None) -> None:
        data = self._fit_data(data, categorical_vars, numerical_vars)
        array, squeezed = _as_2d_numeric(data)
        _require_rows(array)
        _reject_infinite(array)
        check_common_configuration(
            [
                array.shape[1],
                self.n_quantiles,
                self.output_distribution,
                self.n_bins,
                self.max_refinement_rounds,
            ],
            agg_client,
        )
        platform_minimum, configured_minimum = np.asarray(
            agg_client.max([self.platform_minimum, self.min_observations_per_interval]),
            dtype=float,
        )
        effective_minimum = int(max(platform_minimum, configured_minimum))
        global_count = np.asarray(
            agg_client.sum((~np.isnan(array)).sum(axis=0).astype(float)), dtype=float
        )
        required_count = effective_minimum * (self.n_quantiles - 1)
        if np.any(global_count < required_count):
            raise BadInputError(
                "QuantileTransformer requires more global observations per interval."
            )
        references = np.linspace(0.0, 1.0, self.n_quantiles)
        minimum, maximum = _global_min_max(array, agg_client)
        quantiles = []
        diagnostics = []
        for idx in range(array.shape[1]):
            values, diagnostic = batched_quantiles(
                array[:, idx],
                references,
                minimum=minimum[idx],
                maximum=maximum[idx],
                count=int(global_count[idx]),
                minimum_support=effective_minimum,
                n_bins=self.n_bins,
                max_refinement_rounds=self.max_refinement_rounds,
                agg_client=agg_client,
            )
            quantiles.append(values)
            diagnostics.append(diagnostic)
        self.quantiles_ = np.asarray(quantiles, dtype=float)
        self.references_ = references
        self.recipe_ = QuantileTransformerRecipe(
            feature_names=list(self.feature_names_in_),
            categorical_features=list(self.categorical_features_),
            quantiles=self.quantiles_.tolist(),
            references=references.tolist(),
            output_distribution=self.output_distribution,
            min_observations_per_interval=effective_minimum,
            platform_minimum=int(platform_minimum),
            n_bins=self.n_bins,
            max_refinement_rounds=self.max_refinement_rounds,
            fit_diagnostics=diagnostics,
        )
        self.fit_diagnostics_ = self.recipe_.fit_diagnostics
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
        check_fitted(self, "quantiles_", "references_")
        array, squeezed = _as_2d_numeric(data)
        _reject_infinite(array)
        if array.shape[1] != self.quantiles_.shape[0]:
            raise BadInputError(
                "Input feature count does not match the fitted transformer."
            )
        result = np.empty_like(array, dtype=float)
        for idx in range(array.shape[1]):
            if np.all(self.quantiles_[idx] == self.quantiles_[idx, 0]):
                constant_value = 0.5 if self.output_distribution == "uniform" else 0.0
                result[:, idx] = np.where(
                    np.isnan(array[:, idx]), np.nan, constant_value
                )
                continue
            unique_quantiles, inverse = np.unique(
                self.quantiles_[idx], return_inverse=True
            )
            reference_sum = np.bincount(
                inverse, weights=self.references_, minlength=len(unique_quantiles)
            )
            reference_count = np.bincount(inverse, minlength=len(unique_quantiles))
            unique_references = reference_sum / reference_count
            mapped = np.interp(array[:, idx], unique_quantiles, unique_references)
            if self.output_distribution == "normal":
                mapped = ndtri(np.clip(mapped, 1e-7, 1 - 1e-7))
            result[:, idx] = mapped
            result[np.isnan(array[:, idx]), idx] = np.nan
        return _restore_shape(result, squeezed)
