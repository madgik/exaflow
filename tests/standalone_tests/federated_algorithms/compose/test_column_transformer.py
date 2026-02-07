import numpy as np
import pandas as pd

from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder
from exaflow.algorithms.federated.preprocessing import FederatedPassthrough


class DummyAggClient:
    def __init__(self, values_per_call):
        self._values = list(values_per_call)

    def union(self, values):
        return self._values.pop(0)


def test_column_transformer_feature_names_and_shape():
    data = pd.DataFrame(
        {
            "sex": np.array(["M", "F", "M"], dtype=object),
            "age": np.array([10.0, 20.0, 30.0], dtype=float),
        }
    )

    agg_client = DummyAggClient(values_per_call=[["F", "M", "O"]])
    transformer = FederatedColumnTransformer(
        [
            ("cat", FederatedOneHotEncoder(), "categorical"),
            ("num", FederatedPassthrough(), "numerical"),
        ]
    )
    transformer.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )

    feature_names = transformer.get_feature_names_out(
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    assert feature_names == ["sex[M]", "sex[O]", "age"]

    X = transformer.transform(
        data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    assert X.shape == (3, 3)


def test_column_transformer_handles_missing_and_prefix():
    data = pd.DataFrame(
        {
            "sex": np.array(["M", "F"], dtype=object),
            "age": np.array([10.0, 20.0], dtype=float),
        }
    )

    agg_client = DummyAggClient(values_per_call=[["F", "M"]])
    transformer = FederatedColumnTransformer(
        [
            ("cat", FederatedOneHotEncoder(), "categorical"),
            ("num", FederatedPassthrough(), "numerical"),
        ],
        prefix_feature_names=True,
    )
    transformer.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=["sex"],
        numerical_vars=["age", "missing_num"],
    )

    feature_names = transformer.get_feature_names_out(
        categorical_vars=["sex"],
        numerical_vars=["age", "missing_num"],
    )
    assert feature_names == ["cat__sex[M]", "num__age", "num__missing_num"]

    X = transformer.transform(
        data,
        categorical_vars=["sex"],
        numerical_vars=["age", "missing_num"],
    )
    assert X.shape == (2, 3)
