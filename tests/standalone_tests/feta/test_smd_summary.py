import pytest

from exaflow.algorithms.feta._smd import pairwise_smd


def test_pair_order_and_pooled_variance():
    result = pairwise_smd([(2, 4, 3.0, 4.0), (10, 4, 1.0, 4.0)], 2)
    assert result == [{"group1": 10, "group2": 2, "smd": -1.0}]


@pytest.mark.parametrize("minimum", [0, 1, 2, 5])
def test_minimum_count_and_zero_variance(minimum):
    statistics = [("a", 1, 10.0, 0.0), ("b", 4, 20.0, 0.0), ("c", 4, 30.0, 0.0)]
    expected = [] if minimum == 5 else [{"group1": "b", "group2": "c", "smd": 0.0}]
    assert pairwise_smd(statistics, minimum) == expected


def test_no_pairs():
    assert pairwise_smd([], 2) == []
    assert pairwise_smd([("a", 5, 1.0, 1.0)], 2) == []
