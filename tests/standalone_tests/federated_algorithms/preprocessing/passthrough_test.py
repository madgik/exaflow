import numpy as np
import pandas as pd

from exaflow.algorithms.federated.preprocessing import FederatedPassthrough


class DummyAggClient:
    def union(self, values):
        return values


def test_passthrough_handles_missing_and_empty():
    data = pd.DataFrame(
        {
            "age": np.array([10.0, 20.0], dtype=float),
        }
    )
    transformer = FederatedPassthrough()
    transformer.fit(
        agg_client=DummyAggClient(),
        data=data,
        categorical_vars=[],
        numerical_vars=["age", "missing_num"],
    )

    X = transformer.transform(
        data,
        categorical_vars=[],
        numerical_vars=["age", "missing_num"],
    )
    assert X.shape == (2, 2)

    empty_X = transformer.transform(
        data,
        categorical_vars=[],
        numerical_vars=[],
    )
    assert empty_X.shape == (2, 0)
