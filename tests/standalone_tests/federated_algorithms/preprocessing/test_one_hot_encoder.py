import numpy as np
import pandas as pd

from exaflow.algorithms.federated.transformers.one_hot_encoder import (
    FederatedOneHotEncoder,
)


class DummyAggClient:
    def __init__(self, values_per_call):
        self._values = list(values_per_call)

    def union(self, values):
        return self._values.pop(0)


def test_one_hot_encoder_union_and_feature_names():
    data = pd.DataFrame(
        {
            "sex": np.array(["M", "F", "M"], dtype=object),
            "age": np.array([10.0, 20.0, 30.0], dtype=float),
        }
    )

    agg_client = DummyAggClient(values_per_call=[["F", "M", "O"]])
    encoder = FederatedOneHotEncoder()
    encoder.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )

    feature_names = encoder.get_feature_names_out(
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )

    assert feature_names == ["Intercept", "sex[M]", "sex[O]", "age"]
    X = encoder.transform(
        data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    assert X.shape == (3, 4)
