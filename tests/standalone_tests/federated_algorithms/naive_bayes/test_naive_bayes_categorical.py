import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

from exaflow.algorithms.federated.naive_bayes import FederatedCategoricalNB
from tests.standalone_tests.federated_algorithms.utils import FederatedAlgorithmTest

TEST_CASES = [
    (
        pd.DataFrame({"f1": ["A", "B", "A", "B"]}),
        np.array([0, 1, 0, 1]),
        {"f1": ["A", "B"], "y": [0, 1]},
        ["f1"],
    ),
    (
        pd.DataFrame(
            {
                "f1": ["A", "B", "C", "A", "B", "C"],
                "f2": ["X", "Y", "X", "Y", "X", "Y"],
            }
        ),
        np.array([0, 1, 0, 1, 0, 1]),
        {
            "f1": ["A", "B", "C"],
            "f2": ["X", "Y"],
            "y": [0, 1],
        },
        ["f1", "f2"],
    ),
    (
        pd.DataFrame({"f1": ["A", "B", "C", "A", "B", "C"]}),
        np.array([0, 1, 2, 0, 1, 2]),
        {"f1": ["A", "B", "C"], "y": [0, 1, 2]},
        ["f1"],
    ),
    (
        pd.DataFrame(
            {
                "f1": ["A", "B", "C", "A", "B", "C"],
                "f2": ["U", "V", "W", "U", "V", "W"],
            }
        ),
        np.array([0, 1, 2, 0, 1, 2]),
        {
            "f1": ["A", "B", "C"],
            "f2": ["U", "V", "W"],
            "y": [0, 1, 2],
        },
        ["f1", "f2"],
    ),
    (
        pd.DataFrame({"f1": ["A", "B", "A", "B"]}),
        np.array(["yes", "no", "yes", "no"], dtype=object),
        {"f1": ["A", "B"], "y": ["no", "yes"]},
        ["f1"],
    ),
    (
        pd.DataFrame({"f1": ["A", "B", "A", "B"]}),
        np.array([2, 1, 2, 1]),
        {"f1": ["A", "B"], "y": [2, 1]},
        ["f1"],
    ),
    (
        pd.DataFrame({"f1": ["A", "A", "A", "B", "B"]}),
        np.array([1, 1, 1, 0, 0]),
        {"f1": ["A", "B"], "y": [0, 1]},
        ["f1"],
    ),
    (
        pd.DataFrame({"f1": ["A", "B", "C", "D"]}),
        np.array([0, 1, 2, 3]),
        {"f1": ["A", "B", "C", "D"], "y": [0, 1, 2, 3]},
        ["f1"],
    ),
    (
        pd.DataFrame(
            {
                "f1": ["A", "B", "A", "B"],
                "f2": ["X", "X", "Y", "Y"],
                "f3": ["M", "N", "M", "N"],
            }
        ),
        np.array([0, 1, 0, 1]),
        {
            "f1": ["A", "B"],
            "f2": ["X", "Y"],
            "f3": ["M", "N"],
            "y": [0, 1],
        },
        ["f1", "f2", "f3"],
    ),
    (
        pd.DataFrame(
            {
                "f1": ["A", "B", "C", "A", "B", "C"],
                "f2": [1, 2, 3, 1, 2, 3],
            }
        ),
        np.array(["c1", "c2", "c3", "c1", "c2", "c3"], dtype=object),
        {
            "f1": ["A", "B", "C"],
            "f2": [1, 2, 3],
            "y": ["c1", "c2", "c3"],
        },
        ["f1", "f2"],
    ),
]


class TestFederatedCategoricalNB(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        categories = kwargs["categories"]
        x_vars = kwargs["x_vars"]
        encoder = OrdinalEncoder(
            categories=[categories[var] for var in x_vars],
            dtype=int,
        )
        X_enc = encoder.fit_transform(X)
        model = CategoricalNB(alpha=1.0)
        model.fit(X_enc, y)
        return {
            "model": model,
            "X_enc": X_enc,
        }

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        categories = kwargs["categories"]
        x_vars = kwargs["x_vars"]
        encoder = OrdinalEncoder(
            categories=[categories[var] for var in x_vars],
            dtype=int,
        )
        X_enc = encoder.fit_transform(X)
        model = FederatedCategoricalNB(
            y_var="y",
            x_vars=x_vars,
            categories=categories,
        )
        return model.fit(X_enc, y, agg_client=agg_client)

    def compare(self, federated_output, centralized_output, **kwargs):
        model = centralized_output["model"]
        X_enc = centralized_output["X_enc"]
        expected = model.predict(X_enc)
        actual = federated_output.predict(X_enc)
        assert np.array_equal(actual, expected)

    @pytest.mark.parametrize("X, y, categories, x_vars", TEST_CASES)
    def test_federated_algorithm_with_one_worker(self, X, y, categories, x_vars):
        self.run_comparison(X=X, y=y, n_workers=1, categories=categories, x_vars=x_vars)

    @pytest.mark.parametrize("X, y, categories, x_vars", TEST_CASES)
    def test_federated_algorithm_with_multiple_workers(self, X, y, categories, x_vars):
        self.run_comparison(X=X, y=y, n_workers=3, categories=categories, x_vars=x_vars)
