"""Numerical regressions retained through the named-column interface."""

import numpy as np

from exaflow.algorithms.federated.outliers.winsorizer import Winsorizer
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit_in_workers as _fit_in_workers,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)


def test_winsorizer_clips_to_global_fitted_bounds():
    parts = [np.array([[0.0], [1.0], [2.0]]), np.array([[3.0], [4.0], [100.0]])]
    winsorized = _fit_in_workers(lambda: Winsorizer(0.0, 0.8), parts)
    pooled = np.vstack(parts)
    clipped = _transform(winsorized[0], pooled)
    expected = np.clip(pooled, winsorized[0].lower_bounds_, winsorized[0].upper_bounds_)
    np.testing.assert_allclose(clipped, expected)
    assert clipped[-1, 0] == winsorized[0].upper_bounds_[0]
    assert clipped[-1, 0] < pooled[-1, 0]
    assert np.array_equal(parts[0], [[0.0], [1.0], [2.0]])
