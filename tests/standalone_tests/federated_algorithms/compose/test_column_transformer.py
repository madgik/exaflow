import numpy as np
import pandas as pd

from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.compose.column_transformer import make_column_selector
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
        [("cat", FederatedOneHotEncoder(), ["sex"])],
        remainder="passthrough",
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
        [("cat", FederatedOneHotEncoder(), ["sex"])],
        prefix_feature_names=True,
        remainder="passthrough",
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
    assert feature_names == ["cat__sex[M]", "remainder__age", "remainder__missing_num"]

    X = transformer.transform(
        data,
        categorical_vars=["sex"],
        numerical_vars=["age", "missing_num"],
    )
    assert X.shape == (2, 3)


def test_column_transformer_remainder_rejects_categorical():
    data = pd.DataFrame(
        {
            "sex": np.array(["M", "F"], dtype=object),
            "age": np.array([10.0, 20.0], dtype=float),
        }
    )

    agg_client = DummyAggClient(values_per_call=[["F", "M"]])
    transformer = FederatedColumnTransformer(
        [("num", FederatedPassthrough(), ["age"])],
        remainder="passthrough",
    )
    try:
        transformer.fit(
            agg_client=agg_client,
            data=data,
            categorical_vars=["sex"],
            numerical_vars=["age"],
        )
    except ValueError as exc:
        assert "categorical vars" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for categorical remainder passthrough"
        )


def test_column_transformer_selector_callable():
    data = pd.DataFrame(
        {
            "sex": pd.Series(["M", "F"], dtype="category"),
            "age": np.array([10.0, 20.0], dtype=float),
        }
    )

    agg_client = DummyAggClient(values_per_call=[["F", "M"]])
    selector = make_column_selector(dtype_include=["category", "object"])
    transformer = FederatedColumnTransformer(
        [("cat", FederatedOneHotEncoder(), selector)],
        remainder="passthrough",
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
    assert feature_names == ["sex[M]", "age"]


def test_column_transformer_selector_mask_and_indices():
    data = pd.DataFrame(
        {
            "sex": np.array(["M", "F"], dtype=object),
            "age": np.array([10.0, 20.0], dtype=float),
        }
    )

    agg_client = DummyAggClient(values_per_call=[])
    transformer = FederatedColumnTransformer(
        [("num", FederatedPassthrough(), [False, True])],
        remainder="drop",
    )
    transformer.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    X = transformer.transform(
        data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    assert X.shape == (2, 1)

    transformer_idx = FederatedColumnTransformer(
        [("num", FederatedPassthrough(), [1])],
        remainder="drop",
    )
    transformer_idx.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    X_idx = transformer_idx.transform(
        data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    assert X_idx.shape == (2, 1)
