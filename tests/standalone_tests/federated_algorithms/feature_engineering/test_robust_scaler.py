"""Focused robust-scaling regression against pooled reference quantiles."""

import numpy as np

from exaflow.algorithms.federated.feature_engineering import RobustScalerApprox
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit_in_workers as _fit_in_workers,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)


def test_robust_scaler_matches_reference_with_stated_quantile_tolerance():
    pooled = np.linspace(0, 10, 1001).reshape(-1, 1)
    fitted = _fit_in_workers(RobustScalerApprox, [pooled[:100], pooled[100:]])
    q25, median, q75 = np.quantile(pooled[:, 0], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(fitted[0].center_, [median], atol=0.02)
    np.testing.assert_allclose(fitted[0].scale_, [q75 - q25], atol=0.02)
    np.testing.assert_allclose(
        _transform(fitted[1], pooled), (pooled - median) / (q75 - q25), atol=0.02
    )
