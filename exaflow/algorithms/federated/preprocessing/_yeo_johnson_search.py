"""Deterministic, budgeted scalar search over shared Yeo-Johnson objectives."""

import numpy as np
from scipy.optimize import minimize_scalar

from exaflow.algorithms.federated.utils import BadInputError


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
