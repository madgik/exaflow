"""Numerical regressions retained through the named-column interface."""

import numpy as np
import pytest

from exaflow.algorithms.federated.feature_engineering.asinh_transformer import (
    AsinhTransformer,
)
from exaflow.algorithms.federated.feature_engineering.log1p_transformer import (
    Log1pTransformer,
)
from exaflow.algorithms.federated.feature_engineering.sqrt_transformer import (
    SqrtTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit as _fit,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit_in_workers as _fit_in_workers,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


@pytest.mark.parametrize(
    "transformer, expected",
    [
        (Log1pTransformer(), np.log1p([0.0, 1.0, 9.0])),
        (SqrtTransformer(), np.sqrt([0.0, 1.0, 9.0])),
        (AsinhTransformer(), np.arcsinh([-9.0, 0.0, 9.0])),
    ],
)
def test_parameter_transformers_match_numpy_and_do_not_mutate(transformer, expected):
    data = np.array([0.0, 1.0, 9.0])
    if isinstance(transformer, AsinhTransformer):
        data = np.array([-9.0, 0.0, 9.0])
    original = data.copy()
    _fit(transformer, data, agg_client=DummyAggClient())
    assert np.allclose(_transform(transformer, data), expected)
    assert np.array_equal(data, original)


@pytest.mark.parametrize(
    "transformer, invalid",
    [(Log1pTransformer(), np.array([-0.5])), (SqrtTransformer(), np.array([-1.0]))],
)
def test_parameter_transformers_validate_transform_domain(transformer, invalid):
    _fit(transformer, np.array([[0.0]]), agg_client=DummyAggClient())
    with pytest.raises(BadInputError, match="requires non-negative"):
        _transform(transformer, invalid)


@pytest.mark.parametrize("transformer", [Log1pTransformer(), SqrtTransformer()])
def test_domain_validation_is_global(transformer):
    parts = [np.array([[1.0], [2.0]]), np.array([[-1.0], [3.0]])]
    with pytest.raises(BadInputError, match="non-negative"):
        _fit_in_workers(lambda: type(transformer)(), parts)
