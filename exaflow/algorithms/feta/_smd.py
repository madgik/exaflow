"""Cohen's d from globally aggregated group summaries."""

from itertools import combinations

import numpy as np


def pairwise_smd(statistics, minimum_group_size):
    """Return sorted pairwise effects from (label, count, mean, variance) tuples."""
    required_count = max(2, minimum_group_size)
    comparisons = []
    for first, second in combinations(
        sorted(statistics, key=lambda row: str(row[0])), 2
    ):
        group1, n1, mean1, var1 = first
        group2, n2, mean2, var2 = second
        if n1 < required_count or n2 < required_count:
            continue
        pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        effect = 0.0 if pooled <= 0 else float((mean1 - mean2) / np.sqrt(pooled))
        comparisons.append({"group1": group1, "group2": group2, "smd": effect})
    return comparisons
