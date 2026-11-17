import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.utils.positive_class import encode_positive_class
from exaflow.algorithms.exareme3.utils.positive_class import (
    require_binary_positive_class,
)
from exaflow.worker_communication import BadUserInput

METADATA = {
    "event": {"type": "binary", "enumerations": {"0": "Alive", "1": "Dead"}},
    "grade": {"type": "ordinal", "enumerations": {"1": "a", "2": "b"}},
}


@pytest.mark.parametrize("positive_class", ["1", 1])
def test_the_positive_class_maps_to_one_and_the_other_level_to_zero(positive_class):
    events = encode_positive_class(pd.Series(["1", "0", None, "1"]), positive_class)

    np.testing.assert_array_equal(events, np.array([1.0, 0.0, 0.0, 1.0]))


def test_choosing_the_zero_level_flips_the_event():
    events = encode_positive_class(pd.Series(["1", "0", "1"]), "0")

    np.testing.assert_array_equal(events, np.array([0.0, 1.0, 0.0]))


def test_integer_columns_are_compared_after_coercion():
    events = encode_positive_class(pd.Series([1, 2, 3, 2, 1], dtype="int64"), "2")

    np.testing.assert_array_equal(events, np.array([0.0, 1.0, 0.0, 1.0, 0.0]))


def test_a_declared_level_is_accepted():
    require_binary_positive_class(METADATA, "event", "1", role="event variable")


@pytest.mark.parametrize("positive_class", ["2", "Dead", None])
def test_an_undeclared_level_is_rejected(positive_class):
    with pytest.raises(BadUserInput, match="positive_class"):
        require_binary_positive_class(
            METADATA, "event", positive_class, role="event variable"
        )


def test_a_non_binary_variable_is_rejected():
    with pytest.raises(BadUserInput, match="requires a 'binary' variable"):
        require_binary_positive_class(METADATA, "grade", "1", role="event variable")
