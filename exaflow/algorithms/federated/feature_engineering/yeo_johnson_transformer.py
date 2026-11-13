"""YeoJohnsonTransformer: federated fit and named-column transformation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np
from scipy.optimize import minimize_scalar

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


@dataclass(kw_only=True)
class YeoJohnsonFitDiagnostic:
    status: str
    reason: str
    evaluations: int
    searched_bounds: tuple[float, float]
    safe_bounds: tuple[float, float]
    objective: float | None = None

    def __post_init__(self):
        check_choice(
            self.status, ("converged", "approximate", "constant"), name="status"
        )
        check_choice(
            self.reason,
            (
                "convergence",
                "flat_objective",
                "evaluation_budget_exhausted",
                "numerical_bound",
                "constant_feature",
            ),
            name="reason",
        )
        self.evaluations = integer(self.evaluations, name="evaluations")
        ordered_pair(self.safe_bounds, name="safe")
        ordered_pair(self.searched_bounds, name="searched")
        self.safe_bounds = tuple(self.safe_bounds)
        self.searched_bounds = tuple(self.searched_bounds)
        if self.status == "constant":
            if (
                self.reason != "constant_feature"
                or self.evaluations != 0
                or self.objective is not None
            ):
                raise BadInputError(
                    "Constant features do not require search evaluations."
                )
        else:
            if self.evaluations < 1 or self.objective is None:
                raise BadInputError(
                    "A fitted search requires objective and evaluations."
                )
            finite_scalar(self.objective, name="objective")
        if self.status == "converged" and self.reason != "convergence":
            raise BadInputError("Converged fits must report convergence.")
        if self.status == "approximate" and self.reason not in {
            "flat_objective",
            "evaluation_budget_exhausted",
            "numerical_bound",
        }:
            raise BadInputError("Approximate fits must report their limitation.")


@dataclass(kw_only=True)
class YeoJohnsonRecipe(FittedState):
    lambdas: list[float]
    max_evaluations: int
    lambda_tolerance: float
    likelihood_tolerance: float
    fit_diagnostics: list[YeoJohnsonFitDiagnostic]
    primitive_name: str = "yeo_johnson"

    def __post_init__(self):
        self.validate_header(("yeo_johnson",))
        if self.categorical_features:
            raise BadInputError("Yeo-Johnson recipes require numerical features.")
        finite_vector(self.lambdas, name="lambdas", length=len(self.feature_names))
        self.max_evaluations = integer(
            self.max_evaluations, name="max_evaluations", minimum=1
        )
        finite_scalar(
            self.lambda_tolerance,
            name="lambda_tolerance",
            minimum=0,
            include_boundaries="neither",
        )
        finite_scalar(self.likelihood_tolerance, name="likelihood_tolerance", minimum=0)
        if not isinstance(self.fit_diagnostics, list) or len(
            self.fit_diagnostics
        ) != len(self.lambdas):
            raise BadInputError("Provide one search diagnostic per fitted feature.")
        self.fit_diagnostics = [
            load_state(YeoJohnsonFitDiagnostic, d) for d in self.fit_diagnostics
        ]
        for lambd, diagnostic in zip(self.lambdas, self.fit_diagnostics):
            if diagnostic.status == "constant" and lambd != 1:
                raise BadInputError("Constant features must use the identity lambda.")
            if (
                diagnostic.searched_bounds[0] < diagnostic.safe_bounds[0]
                or diagnostic.searched_bounds[1] > diagnostic.safe_bounds[1]
            ):
                raise BadInputError("Recorded search must stay inside numeric bounds.")
            if diagnostic.evaluations > self.max_evaluations:
                raise BadInputError("Fitted search must respect its evaluation budget.")
            if not (
                diagnostic.searched_bounds[0] <= lambd <= diagnostic.searched_bounds[1]
            ):
                raise BadInputError(
                    "Fitted lambda must belong to the recorded search bounds."
                )


def _yeo_johnson(values: np.ndarray, lambd: float) -> np.ndarray:
    if lambd == 1:
        return values.copy()
    result = np.empty_like(values, dtype=float)
    nonnegative = values >= 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if lambd == 0:
            result[nonnegative] = np.log1p(values[nonnegative])
        else:
            result[nonnegative] = (
                np.expm1(lambd * np.log1p(values[nonnegative])) / lambd
            )
        if lambd == 2:
            result[~nonnegative] = -np.log1p(-values[~nonnegative])
        else:
            result[~nonnegative] = -np.expm1(
                (2 - lambd) * np.log1p(-values[~nonnegative])
            ) / (2 - lambd)
    return result


def _yeo_johnson_profile_values(values, lambd, minimum, maximum):
    """Remove only a known additive constant when evaluating variance.

    On one-sided data, saturated power branches contain an additive constant
    that dwarfs the remaining variation. Omitting it leaves variance and the
    profile likelihood unchanged while retaining floating-point information.
    This does not shift the input or alter the actual transform/replay output.
    """
    if minimum >= 0 and lambd * np.log1p(minimum) < -1:
        return np.exp(lambd * np.log1p(values)) / lambd
    power = 2 - lambd
    if maximum <= 0 and power * np.log1p(-maximum) < -1:
        return -np.exp(power * np.log1p(-values)) / power
    return _yeo_johnson(values, lambd)


class _StopSearch(Exception):
    def __init__(self, reason):
        self.reason = reason


def safe_lambda_bounds(minimum, maximum, count):
    """Float64 exponent bounds with headroom for squared sums.

    Follows the exponent-range reasoning used by scipy.stats.yeojohnson_normmax,
    including headroom for observations 20 times the fitted magnitude. Bounds
    use global extrema/counts and therefore agree on every participant.
    """
    finfo = np.finfo(float)
    limit = finfo.max / 4
    lower, upper = -limit, limit
    for magnitude, negative in ((maximum, False), (-minimum, True)):
        if magnitude <= 0:
            continue
        log_magnitude = float(np.logaddexp(0.0, np.log(magnitude) + np.log(20.0)))
        denominator = max(log_magnitude, finfo.tiny)
        with np.errstate(over="ignore"):
            side_lower = (np.log(finfo.tiny) - np.log(finfo.eps)) / (2 * denominator)
            side_upper = (np.log(finfo.max) + np.log(finfo.eps) - np.log(count)) / (
                2 * denominator
            )
        side_lower, side_upper = (
            max(float(side_lower), -limit),
            min(float(side_upper), limit),
        )
        if negative:
            side_lower, side_upper = 2 - side_upper, 2 - side_lower
        lower, upper = max(lower, side_lower), min(upper, side_upper)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise BadInputError("No safe Yeo-Johnson search interval in the numeric range.")
    return lower, upper


def adaptive_lambda_search(
    objective, bounds, *, max_evaluations, lambda_tolerance, likelihood_tolerance
):
    """Return the best finite candidate and an honest termination diagnostic.

    Inputs/objective results must be identical across sites. The objective is
    negative profile log-likelihood per observation. Cached evaluations, a
    shared budget and deterministic tie-breaking keep aggregation calls aligned.
    A converged local search is not a certificate of a global optimum.
    """
    safe_left, safe_right = bounds
    left, right = max(-5.0, safe_left), min(5.0, safe_right)
    if left >= right:
        left, right = safe_left, safe_right
    cache = {}
    history = []

    def evaluate(value):
        value = float(value)
        if value in cache:
            return cache[value]
        if len(cache) >= max_evaluations:
            raise _StopSearch("evaluation_budget_exhausted")
        score = float(objective(value))
        cache[value] = score if np.isfinite(score) else np.inf
        history.append((value, cache[value]))
        return cache[value]

    def best():
        finite = [value for value in cache if np.isfinite(cache[value])]
        if not finite:
            raise BadInputError("No valid Yeo-Johnson fit in the numeric range.")
        return min(finite, key=lambda value: (cache[value], abs(value - 1), value))

    def finish(reason):
        value = best()
        if reason == "flat_objective":
            equivalent = [
                x
                for x, score in cache.items()
                if np.isfinite(score) and score <= cache[value] + likelihood_tolerance
            ]
            value = min(equivalent, key=lambda x: (abs(x - 1), x))
        return value, {
            "status": "converged" if reason == "convergence" else "approximate",
            "reason": reason,
            "evaluations": len(cache),
            "searched_bounds": (min(cache), max(cache)),
            "safe_bounds": bounds,
            "objective": cache[value],
        }

    try:
        # A finite initial candidate allows a useful labelled result even when
        # the caller deliberately supplies a one-evaluation budget.
        evaluate(np.clip(1.0, left, right))
        for point in np.linspace(left, right, 5):
            evaluate(point)
        sampled = [score for score in cache.values() if np.isfinite(score)]
        if len(sampled) >= 4 and np.ptp(sampled) <= likelihood_tolerance:
            return finish("flat_objective")

        while True:
            ordered = sorted(cache)
            candidate = best()
            index = ordered.index(candidate)
            if index == 0:
                new_left = max(safe_left, left - (right - left))
                if new_left == left:
                    return finish("numerical_bound")
                evaluate((new_left + left) / 2)
                evaluate(new_left)
                left = new_left
            elif index == len(ordered) - 1:
                new_right = min(safe_right, right + (right - left))
                if new_right == right:
                    return finish("numerical_bound")
                evaluate((right + new_right) / 2)
                evaluate(new_right)
                right = new_right
            else:
                bracket = (ordered[index - 1], ordered[index + 1])
                break

        def refine(value):
            score = evaluate(value)
            # Several finite, tightly clustered evaluations establish a local
            # likelihood plateau; two nearly equal scores alone do not.
            recent = history[-8:]
            if (
                len(recent) == 8
                and all(np.isfinite(v) for _, v in recent)
                and np.ptp([v for _, v in recent]) <= likelihood_tolerance
                and np.ptp([x for x, _ in recent]) <= 0.01 * (bracket[1] - bracket[0])
            ):
                raise _StopSearch("flat_objective")
            return score

        result = minimize_scalar(
            refine,
            bounds=bracket,
            method="bounded",
            options={"xatol": lambda_tolerance, "maxiter": max_evaluations},
        )
        if not np.isfinite(result.fun):
            return finish("numerical_bound")
        return finish(
            "convergence" if result.success else "evaluation_budget_exhausted"
        )
    except _StopSearch as stopped:
        return finish(stopped.reason)


class YeoJohnsonTransformer(FeatureTransformer):
    """Globally fit a budgeted adaptive power transformation per feature.

    Search starts near [-5, 5], expands when the best candidate is an endpoint,
    and stays inside global float64 exponent bounds. The budget covers all
    objective evaluations per feature, including expansion. Approximate fits
    retain the best finite candidate and explicitly record the stop reason.
    All participants must use identical settings and ordered feature layout.
    """

    def __init__(
        self, *, max_evaluations=64, lambda_tolerance=1e-5, likelihood_tolerance=1e-10
    ):
        if isinstance(max_evaluations, (bool, np.bool_)):
            raise BadInputError("max_evaluations must be an integer.")
        finite_scalar(
            max_evaluations, name="max_evaluations", target_type=Integral, minimum=1
        )
        finite_scalar(
            lambda_tolerance,
            name="lambda_tolerance",
            minimum=0,
            include_boundaries="neither",
        )
        finite_scalar(likelihood_tolerance, name="likelihood_tolerance", minimum=0)
        self.max_evaluations = int(max_evaluations)
        self.lambda_tolerance = float(lambda_tolerance)
        self.likelihood_tolerance = float(likelihood_tolerance)

    @classmethod
    def from_recipe(cls, recipe):
        recipe = load_state(YeoJohnsonRecipe, recipe)
        transformer = cls(
            max_evaluations=recipe.max_evaluations,
            lambda_tolerance=recipe.lambda_tolerance,
            likelihood_tolerance=recipe.likelihood_tolerance,
        )
        transformer.lambdas_ = np.asarray(recipe.lambdas, dtype=float)
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
                self.max_evaluations,
                self.lambda_tolerance,
                self.likelihood_tolerance,
            ],
            agg_client,
        )
        global_min, global_max = _global_min_max(array, agg_client)
        lambdas = []
        diagnostics = []
        for idx in range(array.shape[1]):
            values = array[:, idx]
            valid = values[~np.isnan(values)]
            if global_min[idx] == global_max[idx]:
                lambdas.append(1.0)
                diagnostics.append(
                    {
                        "status": "constant",
                        "reason": "constant_feature",
                        "evaluations": 0,
                        "searched_bounds": (1.0, 1.0),
                        "safe_bounds": (1.0, 1.0),
                        "objective": None,
                    }
                )
                continue

            def objective(lambd):
                # These shared-extrema checks take the same branch everywhere,
                # before feature-specific aggregation. A collapsed output is
                # not a useful fitted transform even if its ideal LL is finite.
                try:
                    endpoints = _yeo_johnson(
                        np.array([global_min[idx], global_max[idx]]), lambd
                    )
                except FloatingPointError:
                    return np.inf
                if endpoints[0] == endpoints[1]:
                    return np.inf
                local_error = False
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        transformed = _yeo_johnson_profile_values(
                            valid, float(lambd), global_min[idx], global_max[idx]
                        )
                        local_sum = transformed.sum()
                except FloatingPointError:
                    local_error = True
                    local_sum = 0.0
                total_sum, total_error = np.asarray(
                    agg_client.sum([local_sum, float(local_error)]), dtype=float
                )
                if total_error or not np.isfinite(total_sum):
                    return np.inf
                mean = total_sum / total_n
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        differences = transformed - mean
                        local_squared = np.square(differences).sum()
                        local_correction = differences.sum()
                except FloatingPointError:
                    local_error = True
                    local_squared = 0.0
                    local_correction = 0.0
                total_squared, correction, total_error = np.asarray(
                    agg_client.sum(
                        [local_squared, local_correction, float(local_error)]
                    ),
                    dtype=float,
                )
                if total_error or not np.isfinite(total_squared):
                    return np.inf
                variance = (
                    total_squared - correction * (correction / total_n)
                ) / total_n
                if variance <= 0 or not np.isfinite(variance):
                    return np.inf
                return 0.5 * np.log(variance) - (lambd - 1) * (total_jacobian / total_n)

            jacobian = np.sign(valid) * np.log1p(np.abs(valid))
            total_n, total_jacobian = np.asarray(
                agg_client.sum([len(valid), jacobian.sum()]), dtype=float
            )
            bounds = safe_lambda_bounds(global_min[idx], global_max[idx], total_n)
            lambd, diagnostic = adaptive_lambda_search(
                objective,
                bounds,
                max_evaluations=self.max_evaluations,
                lambda_tolerance=self.lambda_tolerance,
                likelihood_tolerance=self.likelihood_tolerance,
            )
            lambdas.append(lambd)
            diagnostics.append(diagnostic)
        self.lambdas_ = np.asarray(lambdas)
        self.recipe_ = YeoJohnsonRecipe(
            feature_names=list(self.feature_names_in_),
            categorical_features=list(self.categorical_features_),
            lambdas=self.lambdas_.tolist(),
            max_evaluations=self.max_evaluations,
            lambda_tolerance=self.lambda_tolerance,
            likelihood_tolerance=self.likelihood_tolerance,
            fit_diagnostics=diagnostics,
        )
        self.fit_diagnostics_ = self.recipe_.fit_diagnostics
        return None

    def transform(self, data, *, categorical_vars, numerical_vars=None) -> np.ndarray:
        data = self._transform_data(data, categorical_vars, numerical_vars)
        check_fitted(self, "lambdas_")
        array, squeezed = _as_2d_numeric(data)
        _reject_infinite(array)
        if array.shape[1] != len(self.lambdas_):
            raise BadInputError(
                "Input feature count does not match the fitted transformer."
            )
        result = np.empty_like(array, dtype=float)
        for idx, lambd in enumerate(self.lambdas_):
            valid = ~np.isnan(array[:, idx])
            result[:, idx] = np.nan
            try:
                result[valid, idx] = _yeo_johnson(array[valid, idx], float(lambd))
            except FloatingPointError:
                raise BadInputError(
                    "Yeo-Johnson transformation is undefined for the input."
                ) from None
        return _restore_shape(result, squeezed)
