from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SUPPRESSED_CLUSTER_SIZE_LABEL = "suppressed"


@dataclass(frozen=True)
class ClusterCountPrivacy:
    """
    Privacy view for one cluster count.

    `exact_count` must remain internal. Public API results should use
    `size_interval` and `can_show_profile`.
    """

    size_interval: str
    can_show_profile: bool


def mask_cluster_counts(
    counts: Iterable[int],
    *,
    minimum_row_count: int,
) -> list[ClusterCountPrivacy]:
    return [
        mask_cluster_count(count, minimum_row_count=minimum_row_count)
        for count in counts
    ]


def mask_cluster_count(
    count: int,
    *,
    minimum_row_count: int,
) -> ClusterCountPrivacy:
    count = int(count)
    minimum_row_count = int(minimum_row_count)
    if minimum_row_count <= 0:
        raise ValueError("minimum_row_count must be positive.")

    if count == 0:
        return ClusterCountPrivacy(
            size_interval="0",
            can_show_profile=False,
        )
    if count < minimum_row_count:
        return ClusterCountPrivacy(
            size_interval=f"<{minimum_row_count}",
            can_show_profile=False,
        )

    lower = (count // minimum_row_count) * minimum_row_count
    upper = lower + minimum_row_count - 1
    return ClusterCountPrivacy(
        size_interval=f"{lower}-{upper}",
        can_show_profile=True,
    )


def validate_binary_cluster_privacy(
    *,
    selected_count: int,
    other_count: int,
    minimum_row_count: int,
) -> None:
    selected = mask_cluster_count(
        selected_count,
        minimum_row_count=minimum_row_count,
    )
    other = mask_cluster_count(
        other_count,
        minimum_row_count=minimum_row_count,
    )
    if not selected.can_show_profile or not other.can_show_profile:
        raise ValueError(
            "Cannot create binary KMeans cluster variable because one class is "
            "below the privacy threshold."
        )
