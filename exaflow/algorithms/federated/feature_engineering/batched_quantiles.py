"""Shared histogram refinement with global rank offsets and support guards."""

from dataclasses import dataclass

import numpy as np

from exaflow.algorithms.federated.utils import BadInputError


@dataclass
class _Node:
    values: np.ndarray
    targets: list[int]
    before: int
    count: int
    bounds: tuple[float, float] | None = None


def _extrema(nodes, agg_client):
    """Only request extrema for groups already known to meet minimum support."""
    if all(node.bounds is not None for node in nodes):
        return [node.bounds for node in nodes]
    lower = np.asarray(
        agg_client.min([np.min(node.values, initial=np.inf) for node in nodes]),
        dtype=float,
    )
    upper = np.asarray(
        agg_client.max([np.max(node.values, initial=-np.inf) for node in nodes]),
        dtype=float,
    )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise BadInputError("Refined quantile groups must have global observations.")
    return list(zip(lower, upper))


def _supported_window(counts, prefix, target_bin, minimum_support):
    """Expand a target's bin using nearest occupied neighbours until supported.

    Windows for different quantiles may overlap. Each remains an independent
    conditional histogram with its own global offset; counts from different
    windows are never added together. Identical windows share one next node.
    """
    occupied = np.flatnonzero(counts)
    position = int(np.searchsorted(occupied, target_bin))
    left = right = position
    while prefix[occupied[right] + 1] - prefix[occupied[left]] < minimum_support:
        left_cost = (occupied[left] - occupied[left - 1]) if left else np.inf
        right_cost = (
            (occupied[right + 1] - occupied[right])
            if right + 1 < len(occupied)
            else np.inf
        )
        if left_cost <= right_cost:
            left -= 1
        else:
            right += 1
    return int(occupied[left]), int(occupied[right])


def batched_quantiles(
    values,
    references,
    *,
    minimum,
    maximum,
    count,
    minimum_support,
    n_bins,
    max_refinement_rounds,
    agg_client,
):
    """Fit all target quantiles with one packed count exchange per level.

    Each interior target retains a group containing at least minimum_support
    global observations. No extrema are requested for smaller groups. Counts
    still reveal histogram information; this is not differential privacy.
    The initial histogram is followed by at most max_refinement_rounds levels.
    Two additional vector exchanges trim each selected group's observed range.
    Local membership follows the exact bins used for counting, avoiding loss or
    duplication of edge observations after floating-point coordinate changes.
    """
    with np.errstate(over="ignore"):
        span = maximum - minimum
    if not np.isfinite(span):
        raise BadInputError("Quantile histogram range exceeds the numeric range.")
    n_targets = len(references)
    estimates = np.linspace(minimum, maximum, n_targets)
    intervals = np.tile([minimum, maximum], (n_targets, 1)).astype(float)
    intervals[0] = minimum
    intervals[-1] = maximum
    support = np.zeros(n_targets, dtype=int)
    nodes = (
        [
            _Node(
                values[~np.isnan(values)],
                list(range(1, n_targets - 1)),
                0,
                int(count),
                (minimum, maximum),
            )
        ]
        if n_targets > 2
        else []
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    histogram_rounds = 0
    minimum_limited = False
    budget_limited = False

    def assign(node, bounds):
        lower, upper = bounds
        targets = node.targets
        fractions = (references[targets] * count - node.before) / node.count
        estimates[targets] = lower + np.clip(fractions, 0, 1) * (upper - lower)
        intervals[targets] = [lower, upper]
        support[targets] = node.count

    for _ in range(max_refinement_rounds + 1):
        if not nodes:
            break
        bounds = _extrema(nodes, agg_client)
        active = []
        for node, (lower, upper) in zip(nodes, bounds):
            if lower == upper:
                assign(node, (lower, upper))
            elif node.count == minimum_support:
                assign(node, (lower, upper))
                minimum_limited = True
            else:
                active.append((node, lower, upper))
        if not active:
            nodes = []
            break

        local_bins = []
        local_counts = []
        for node, lower, upper in active:
            normalized = (node.values - lower) / (upper - lower)
            bin_ids = np.clip(
                np.searchsorted(edges, normalized, side="right") - 1, 0, n_bins - 1
            )
            local_bins.append(bin_ids)
            local_counts.append(np.bincount(bin_ids, minlength=n_bins).astype(float))
        packed = np.asarray(agg_client.sum(np.concatenate(local_counts)), dtype=float)
        counts_by_node = packed.reshape(len(active), n_bins)
        histogram_rounds += 1
        next_nodes = []

        for (node, lower, upper), bin_ids, counts in zip(
            active, local_bins, counts_by_node
        ):
            if (
                not np.isfinite(counts).all()
                or np.any(counts < 0)
                or np.sum(counts) != node.count
            ):
                raise BadInputError(
                    "Refined histogram counts must preserve global ranks."
                )
            prefix = np.r_[0.0, np.cumsum(counts)]
            ranks = references[node.targets] * count - node.before
            target_bins = np.minimum(
                np.searchsorted(prefix[1:], ranks, side="left"), n_bins - 1
            )
            windows = {}
            for target, target_bin in zip(node.targets, target_bins):
                window = _supported_window(counts, prefix, target_bin, minimum_support)
                windows.setdefault(window, []).append(target)
            for (left, right), targets in windows.items():
                child_count = int(prefix[right + 1] - prefix[left])
                if child_count == node.count:
                    # Repeating the same histogram cannot narrow this group
                    # without violating the configured support floor.
                    assign(
                        _Node(node.values, targets, node.before, node.count),
                        (lower, upper),
                    )
                    minimum_limited = True
                    continue
                selected = (bin_ids >= left) & (bin_ids <= right)
                next_nodes.append(
                    _Node(
                        node.values[selected],
                        targets,
                        node.before + int(prefix[left]),
                        child_count,
                    )
                )
        nodes = next_nodes

    if nodes:
        for node, bounds in zip(nodes, _extrema(nodes, agg_client)):
            assign(node, bounds)
            if bounds[0] != bounds[1]:
                minimum_limited |= node.count == minimum_support
                budget_limited |= node.count > minimum_support

    # Different, overlapping supported windows can interpolate out of order.
    # Project onto a monotone sequence while retaining every certified interval.
    feasible_lower = np.maximum.accumulate(intervals[:, 0])
    feasible_upper = np.minimum.accumulate(intervals[::-1, 1])[::-1]
    if np.any(feasible_lower > feasible_upper):
        raise BadInputError("Refined quantile intervals must preserve rank ordering.")
    estimates = np.maximum.accumulate(
        np.clip(estimates, feasible_lower, feasible_upper)
    )
    diagnostic = {
        "histogram_rounds": histogram_rounds,
        "intervals": intervals.tolist(),
        "support_counts": support.tolist(),
        "max_interval_width": float(np.max(intervals[:, 1] - intervals[:, 0])),
        "limited_by_minimum_support": bool(minimum_limited),
        "limited_by_refinement_budget": bool(budget_limited),
    }
    return estimates, diagnostic
