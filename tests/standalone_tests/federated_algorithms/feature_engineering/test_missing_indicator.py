"""Numerical regressions retained through the named-column interface."""

import numpy as np

from exaflow.algorithms.federated.feature_engineering.missing_indicator import (
    MissingIndicator,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit as _fit,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)


def test_missing_indicator_is_parameter_free_and_preserves_shape():
    data = np.array([[1.0, np.nan], [np.nan, 2.0]])
    original = data.copy()
    indicator = _fit(MissingIndicator(), data)
    assert _transform(indicator, data).tolist() == [[0, 1], [1, 0]]
    assert np.array_equal(data, original, equal_nan=True)
    assert _transform(
        _fit(MissingIndicator(), np.array([1.0, np.nan])), np.array([1.0, np.nan])
    ).tolist() == [0, 1]
