"""Numerical regressions retained through the named-column interface."""

import numpy as np

from exaflow.algorithms.federated.outliers.outlier_detector import OutlierDetector
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit as _fit,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


def test_outlier_detector_treats_rounding_only_scale_as_constant():
    data = np.array([[0.1], [0.1], [0.1]])
    detector = _fit(OutlierDetector(), data, agg_client=DummyAggClient())
    assert not _transform(detector, data).any()


def test_outlier_detector_is_diagnostic_only():
    data = np.array([[0.0], [0.0], [0.0], [10.0]])
    detector = _fit(OutlierDetector(threshold=1.5), data, agg_client=DummyAggClient())
    mask = _transform(detector, data)
    assert mask.dtype == bool
    assert mask[-1]
    assert np.array_equal(data, [[0.0], [0.0], [0.0], [10.0]])
