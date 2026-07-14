import pytest

from exaflow.algorithms.federated.cluster.kmeans_privacy import mask_cluster_count
from exaflow.algorithms.federated.cluster.kmeans_privacy import mask_cluster_counts
from exaflow.algorithms.federated.cluster.kmeans_privacy import (
    validate_binary_cluster_privacy,
)


def test_mask_cluster_count_suppresses_empty_and_small_clusters():
    empty = mask_cluster_count(0, minimum_row_count=10)
    small = mask_cluster_count(7, minimum_row_count=10)

    assert empty.size_interval == "0"
    assert empty.can_show_profile is False
    assert small.size_interval == "<10"
    assert small.can_show_profile is False


@pytest.mark.parametrize(
    "count, expected_interval",
    [
        (10, "10-19"),
        (19, "10-19"),
        (20, "20-29"),
        (57, "50-59"),
    ],
)
def test_mask_cluster_count_returns_intervals_for_public_counts(
    count,
    expected_interval,
):
    masked = mask_cluster_count(count, minimum_row_count=10)

    assert masked.size_interval == expected_interval
    assert masked.can_show_profile is True


def test_mask_cluster_counts_masks_each_cluster():
    masked = mask_cluster_counts([0, 4, 10, 23], minimum_row_count=10)

    assert [item.size_interval for item in masked] == [
        "0",
        "<10",
        "10-19",
        "20-29",
    ]
    assert [item.can_show_profile for item in masked] == [
        False,
        False,
        True,
        True,
    ]


def test_mask_cluster_count_rejects_invalid_minimum_row_count():
    with pytest.raises(ValueError, match="minimum_row_count"):
        mask_cluster_count(10, minimum_row_count=0)


def test_validate_binary_cluster_privacy_accepts_two_public_classes():
    validate_binary_cluster_privacy(
        selected_count=10,
        other_count=100,
        minimum_row_count=10,
    )


@pytest.mark.parametrize(
    "selected_count, other_count",
    [
        (9, 100),
        (10, 9),
        (0, 100),
    ],
)
def test_validate_binary_cluster_privacy_rejects_small_classes(
    selected_count,
    other_count,
):
    with pytest.raises(ValueError, match="binary KMeans cluster variable"):
        validate_binary_cluster_privacy(
            selected_count=selected_count,
            other_count=other_count,
            minimum_row_count=10,
        )
