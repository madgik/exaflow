import numpy as np
import pandas as pd

from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.linear_model.ols import FederatedOLS
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


def test_pipeline_fit_transform_and_estimator():
    data = pd.DataFrame(
        {
            "sex": np.array(["M", "F", "M"], dtype=object),
            "age": np.array([10.0, 20.0, 30.0], dtype=float),
        }
    )

    agg_client = DummyAggClient()
    column_transformer = FederatedColumnTransformer(
        [("cat", FederatedOneHotEncoder(), ["sex"])],
        remainder="passthrough",
    )
    pipeline = FederatedPipeline(
        [
            ("features", column_transformer),
            ("model", FederatedOLS(fit_intercept=True)),
        ]
    )
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    results = pipeline.fit(
        agg_client=agg_client,
        data=data,
        y=y,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    assert results is not None

    X = pipeline.transform(
        data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    assert X.shape == (3, 2)


def test_pipeline_transformers_only():
    data = pd.DataFrame(
        {
            "sex": np.array(["M", "F"], dtype=object),
            "age": np.array([10.0, 20.0], dtype=float),
        }
    )

    agg_client = DummyAggClient()
    column_transformer = FederatedColumnTransformer(
        [("cat", FederatedOneHotEncoder(), ["sex"])],
        remainder="passthrough",
    )
    pipeline = FederatedPipeline([("features", column_transformer)])
    pipeline.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )

    X = pipeline.transform(
        data,
        categorical_vars=["sex"],
        numerical_vars=["age"],
    )
    assert X.shape == (2, 2)
