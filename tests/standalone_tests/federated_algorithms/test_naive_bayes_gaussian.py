import numpy as np
from sklearn.naive_bayes import GaussianNB

from exaflow.algorithms.federated.naive_bayes_gaussian import FederatedGaussianNB
from tests.standalone_tests.federated_algorithms.utils import DummyAggClient


def _fit_predict_federated(X, y, labels):
    agg_client = DummyAggClient()
    model = FederatedGaussianNB(
        x_vars=[f"x{i}" for i in range(X.shape[1])], labels=labels
    )
    results = model.fit(X, y, agg_client=agg_client)
    return results.predict(X)


def _fit_predict_sklearn(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model.predict(X)


def _run_case(X, y):
    labels = sorted(np.unique(y).tolist())
    y_pred_fed = _fit_predict_federated(X, y, labels)
    y_pred_skl = _fit_predict_sklearn(X, y)
    assert y_pred_fed.tolist() == y_pred_skl.tolist()


def test_gaussian_nb_binary_single_feature():
    X = np.array([[0.0], [1.0], [0.1], [0.9]])
    y = np.array([0, 1, 0, 1])
    _run_case(X, y)


def test_gaussian_nb_binary_two_features():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ]
    )
    y = np.array([0, 1, 0, 1])
    _run_case(X, y)


def test_gaussian_nb_multiclass_single_feature():
    X = np.array([[0.0], [1.0], [2.0], [0.1], [1.1], [2.1]])
    y = np.array([0, 1, 2, 0, 1, 2])
    _run_case(X, y)


def test_gaussian_nb_multiclass_two_features():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.1, 0.1],
            [1.1, 1.1],
            [2.1, 2.1],
        ]
    )
    y = np.array([0, 1, 2, 0, 1, 2])
    _run_case(X, y)


def test_gaussian_nb_imbalanced_classes():
    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1]])
    y = np.array([0, 0, 0, 1, 1])
    _run_case(X, y)
