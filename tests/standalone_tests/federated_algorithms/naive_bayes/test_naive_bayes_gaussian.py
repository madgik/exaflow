import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

from exaflow.algorithms.federated.naive_bayes import FederatedGaussianNB
from tests.standalone_tests.federated_algorithms.utils import FederatedAlgorithmTest

TEST_CASES = [
    (np.array([[0.0], [1.0], [0.1], [0.9]]), np.array([0, 1, 0, 1])),
    (
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.2, 0.8],
                [0.8, 0.2],
            ]
        ),
        np.array([0, 1, 0, 1]),
    ),
    (
        np.array([[0.0], [1.0], [2.0], [0.1], [1.1], [2.1]]),
        np.array([0, 1, 2, 0, 1, 2]),
    ),
    (
        np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [0.1, 0.1],
                [1.1, 1.1],
                [2.1, 2.1],
            ]
        ),
        np.array([0, 1, 2, 0, 1, 2]),
    ),
    (np.array([[0.0], [0.1], [0.2], [1.0], [1.1]]), np.array([0, 0, 0, 1, 1])),
]


class TestFederatedGaussianNB(FederatedAlgorithmTest):
    def compute_centralized_result(self, X, y, **kwargs):
        model = GaussianNB()
        model.fit(X, y)
        return {
            "model": model,
            "X": X,
        }

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        labels = kwargs["labels"]
        model = FederatedGaussianNB(
            x_vars=[f"x{i}" for i in range(X.shape[1])], labels=labels
        )
        return model.fit(X, y, agg_client=agg_client)

    def compare(self, federated_output, centralized_output, **kwargs):
        model = centralized_output["model"]
        X = centralized_output["X"]
        expected = model.predict(X)
        actual = federated_output.predict(X)
        assert np.array_equal(actual, expected)

    @pytest.mark.parametrize("X, y", TEST_CASES)
    def test_federated_algorithm_with_one_worker(self, X, y):
        labels = sorted(np.unique(y).tolist())
        self.run_comparison(X=X, y=y, n_workers=1, labels=labels)

    @pytest.mark.parametrize("X, y", TEST_CASES)
    def test_federated_algorithm_with_multiple_workers(self, X, y):
        labels = sorted(np.unique(y).tolist())
        self.run_comparison(X=X, y=y, n_workers=3, labels=labels)
