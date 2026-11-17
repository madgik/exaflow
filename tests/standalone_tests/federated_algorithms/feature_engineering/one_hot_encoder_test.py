import numpy as np
import pandas as pd

from exaflow.algorithms.federated.feature_engineering._one_hot_encoder import (
    _FederatedOneHotEncoder,
)
from exaflow.algorithms.federated.utils import sort_levels


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
    encoder = _FederatedOneHotEncoder()
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

    assert feature_names == ["sex[M]", "sex[O]", "age"]
    X = encoder.transform(
        data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    assert X.shape == (3, 3)


def test_levels_sort_numerically_and_one_hot_drops_the_lowest_code():
    assert sort_levels(["10", "5", "2"]) == ["2", "5", "10"]
    assert sort_levels(["b", "a"]) == ["a", "b"]

    levels = [str(code) for code in range(5, 13)]
    data = pd.DataFrame({"var": pd.Series(levels, dtype=object)})
    agg_client = DummyAggClient(values_per_call=[levels])
    encoder = _FederatedOneHotEncoder()
    encoder.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=["var"],
        numerical_vars=[],
    )

    assert encoder.get_feature_names_out(
        categorical_vars=["var"], numerical_vars=[]
    ) == [f"var[{code}]" for code in range(6, 13)]
