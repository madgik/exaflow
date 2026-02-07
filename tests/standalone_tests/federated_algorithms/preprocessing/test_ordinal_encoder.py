import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.transformers.ordinal_encoder import (
    FederatedOrdinalEncoder,
)


class DummyAggClient:
    def sum(self, values):
        return values


def test_ordinal_encoder_strict_unknown_raises():
    data = pd.DataFrame({"sex": np.array(["M", "F"], dtype=object)})
    categories = {"sex": ["F", "M"]}

    encoder = FederatedOrdinalEncoder(
        categories=categories,
        handle_unknown="error",
    )
    encoder.fit(
        agg_client=DummyAggClient(),
        data=data,
        categorical_vars=["sex"],
    )

    bad = pd.DataFrame({"sex": np.array(["F", "X"], dtype=object)})
    with pytest.raises(ValueError, match="Unknown categories"):
        encoder.transform(
            bad,
            categorical_vars=["sex"],
            numerical_vars=[],
        )


def test_ordinal_encoder_unknown_maps_to_default():
    data = pd.DataFrame({"sex": np.array(["M", "F"], dtype=object)})
    categories = {"sex": ["F", "M"]}

    encoder = FederatedOrdinalEncoder(
        categories=categories,
        handle_unknown="ignore",
        unknown_value=-1,
    )
    encoder.fit(
        agg_client=DummyAggClient(),
        data=data,
        categorical_vars=["sex"],
    )

    X = encoder.transform(
        pd.DataFrame({"sex": np.array(["F", "X"], dtype=object)}),
        categorical_vars=["sex"],
        numerical_vars=[],
    )
    assert X.tolist() == [[0], [-1]]


def test_ordinal_encoder_missing_column_uses_unknown_value():
    data = pd.DataFrame({"sex": np.array(["M", "F"], dtype=object)})
    categories = {"sex": ["F", "M"]}

    encoder = FederatedOrdinalEncoder(categories=categories)
    encoder.fit(
        agg_client=DummyAggClient(),
        data=data,
        categorical_vars=["sex"],
    )

    X = encoder.transform(
        pd.DataFrame({"other": np.array(["A", "B"], dtype=object)}),
        categorical_vars=["sex"],
        numerical_vars=[],
    )
    assert X.shape == (2, 1)
    assert np.all(X == -1)
