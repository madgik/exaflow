import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

from exaflow.algorithms.federated.naive_bayes import FederatedCategoricalNB
from tests.standalone_tests.federated_algorithms.utils import DummyAggClient


def _fit_predict_federated(X_enc, y, categories, x_vars, y_var="y"):
    agg_client = DummyAggClient()
    model = FederatedCategoricalNB(
        y_var=y_var,
        x_vars=x_vars,
        categories=categories,
    )
    results = model.fit(X_enc, y, agg_client=agg_client)
    return results.predict(X_enc)


def _fit_predict_sklearn(X_enc, y):
    model = CategoricalNB(alpha=1.0)
    model.fit(X_enc, y)
    return model.predict(X_enc)


def _encode_features(X_raw, categories, x_vars):
    encoder = OrdinalEncoder(
        categories=[categories[var] for var in x_vars],
        dtype=int,
    )
    return encoder.fit_transform(X_raw)


def _run_case(X_raw, y, categories, x_vars):
    X_enc = _encode_features(X_raw, categories, x_vars)
    y_pred_fed = _fit_predict_federated(X_enc, y, categories, x_vars)
    y_pred_skl = _fit_predict_sklearn(X_enc, y)
    assert y_pred_fed.tolist() == y_pred_skl.tolist()


def test_nb_categorical_binary_single_feature():
    X = pd.DataFrame({"f1": ["A", "B", "A", "B"]})
    y = np.array([0, 1, 0, 1])
    categories = {"f1": ["A", "B"], "y": [0, 1]}
    _run_case(X, y, categories, ["f1"])


def test_nb_categorical_binary_two_features():
    X = pd.DataFrame(
        {
            "f1": ["A", "B", "C", "A", "B", "C"],
            "f2": ["X", "Y", "X", "Y", "X", "Y"],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])
    categories = {
        "f1": ["A", "B", "C"],
        "f2": ["X", "Y"],
        "y": [0, 1],
    }
    _run_case(X, y, categories, ["f1", "f2"])


def test_nb_categorical_multiclass_single_feature():
    X = pd.DataFrame({"f1": ["A", "B", "C", "A", "B", "C"]})
    y = np.array([0, 1, 2, 0, 1, 2])
    categories = {"f1": ["A", "B", "C"], "y": [0, 1, 2]}
    _run_case(X, y, categories, ["f1"])


def test_nb_categorical_multiclass_two_features():
    X = pd.DataFrame(
        {
            "f1": ["A", "B", "C", "A", "B", "C"],
            "f2": ["U", "V", "W", "U", "V", "W"],
        }
    )
    y = np.array([0, 1, 2, 0, 1, 2])
    categories = {
        "f1": ["A", "B", "C"],
        "f2": ["U", "V", "W"],
        "y": [0, 1, 2],
    }
    _run_case(X, y, categories, ["f1", "f2"])


def test_nb_categorical_string_labels():
    X = pd.DataFrame({"f1": ["A", "B", "A", "B"]})
    y = np.array(["yes", "no", "yes", "no"], dtype=object)
    categories = {"f1": ["A", "B"], "y": ["no", "yes"]}
    _run_case(X, y, categories, ["f1"])


def test_nb_categorical_non_sorted_int_labels():
    X = pd.DataFrame({"f1": ["A", "B", "A", "B"]})
    y = np.array([2, 1, 2, 1])
    categories = {"f1": ["A", "B"], "y": [2, 1]}
    _run_case(X, y, categories, ["f1"])


def test_nb_categorical_imbalanced_classes():
    X = pd.DataFrame({"f1": ["A", "A", "A", "B", "B"]})
    y = np.array([1, 1, 1, 0, 0])
    categories = {"f1": ["A", "B"], "y": [0, 1]}
    _run_case(X, y, categories, ["f1"])


def test_nb_categorical_four_classes():
    X = pd.DataFrame({"f1": ["A", "B", "C", "D"]})
    y = np.array([0, 1, 2, 3])
    categories = {"f1": ["A", "B", "C", "D"], "y": [0, 1, 2, 3]}
    _run_case(X, y, categories, ["f1"])


def test_nb_categorical_three_features_binary():
    X = pd.DataFrame(
        {
            "f1": ["A", "B", "A", "B"],
            "f2": ["X", "X", "Y", "Y"],
            "f3": ["M", "N", "M", "N"],
        }
    )
    y = np.array([0, 1, 0, 1])
    categories = {
        "f1": ["A", "B"],
        "f2": ["X", "Y"],
        "f3": ["M", "N"],
        "y": [0, 1],
    }
    _run_case(X, y, categories, ["f1", "f2", "f3"])


def test_nb_categorical_multiclass_mixed_types():
    X = pd.DataFrame(
        {
            "f1": ["A", "B", "C", "A", "B", "C"],
            "f2": [1, 2, 3, 1, 2, 3],
        }
    )
    y = np.array(["c1", "c2", "c3", "c1", "c2", "c3"], dtype=object)
    categories = {
        "f1": ["A", "B", "C"],
        "f2": [1, 2, 3],
        "y": ["c1", "c2", "c3"],
    }
    _run_case(X, y, categories, ["f1", "f2"])
