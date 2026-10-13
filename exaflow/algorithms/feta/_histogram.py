"""Owned histogram boundary sizing and count-release helpers."""

import math
from typing import Tuple


def _wilkinson_step(
    data_min: float, data_max: float, num_bins: int, *, min_step: float = 0.0
) -> Tuple[float, float, float, int]:
    """Wilkinson nice-numbers bin sizing.

    Snaps the bin width to the nearest 1/2/5 × 10^n, then extends min/max
    outward to the nearest step boundary.

    Args:
        min_step: Floor on the resulting step. Pass 1.0 for integer-valued
            data so a single bin never splits two adjacent integers apart.
            When this floor brings the step down to exactly 1, bin edges are
            aligned to whole numbers (one bin per integer value).

    Returns:
        (step, new_min, new_max, new_num_bins)
    """
    raw_step = (data_max - data_min) / num_bins
    if raw_step == 0:
        step = max(1.0, min_step)
        if min_step > 0 and step == 1.0:
            new_min = math.floor(data_min)
            new_max = math.ceil(data_max) + 1
            return step, float(new_min), float(new_max), int(new_max - new_min)
        half = step / 2
        return step, data_min - half, data_min + half, 1
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = min([1, 2, 5], key=lambda m: abs(raw_step - m * magnitude)) * magnitude
    step = max(step, min_step)

    if min_step > 0 and step == 1.0:
        new_min = math.floor(data_min)
        new_max = math.ceil(data_max) + 1
        new_num_bins = int(new_max - new_min)
        return step, float(new_min), float(new_max), new_num_bins

    new_min = math.floor(data_min / step) * step
    new_max = math.ceil(data_max / step) * step
    new_num_bins = round((new_max - new_min) / step)
    return step, new_min, new_max, new_num_bins


def mask_counts(values, minimum):
    result = []
    for value in values:
        count = int(round(float(value)))
        result.append(count if count >= minimum else None)
    return result
