"""The one-hot and ordinal encoders are private building blocks of the algorithms."""

import importlib

import pytest

import exaflow.algorithms.federated as federated
import exaflow.algorithms.federated.feature_engineering as feature_engineering
from exaflow import exareme3_algorithm_classes
from exaflow import exareme3_preprocessing_step_classes


@pytest.mark.parametrize(
    "module",
    [
        "exaflow.algorithms.federated.feature_engineering.one_hot_encoder",
        "exaflow.algorithms.federated.feature_engineering.ordinal_encoder",
    ],
)
def test_public_encoder_modules_no_longer_exist(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_encoders_are_not_exported():
    assert feature_engineering.__all__ == []
    for name in ("FederatedOneHotEncoder", "FederatedOrdinalEncoder"):
        assert not hasattr(federated, name)
        assert name not in federated.__all__
        assert not hasattr(feature_engineering, name)


def test_encoders_are_not_algorithms_or_preprocessing_steps():
    published = {
        *exareme3_algorithm_classes,
        *exareme3_preprocessing_step_classes,
    }
    assert not {name for name in published if "encoder" in name or "one_hot" in name}
