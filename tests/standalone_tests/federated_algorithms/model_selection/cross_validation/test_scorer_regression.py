from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from exaflow.algorithms.federated.model_selection import FederatedRegressionScorer
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    _simulate_federated_execution,
)


def score(y, predictions, *, n_workers=1, p=1, n_train=20):
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    indices = np.array_split(np.arange(len(y)), n_workers)

    def worker(worker_id, agg_client):
        idx = indices[worker_id]
        results = SimpleNamespace(predict=lambda X: predictions[idx])
        return FederatedRegressionScorer().score(
            results,
            np.empty((len(idx), p)),
            y[idx],
            agg_client=agg_client,
            n_train=n_train,
            p=p,
        )

    outputs = _simulate_federated_execution(n_workers, worker)
    for output in outputs[1:]:
        np.testing.assert_allclose(
            list(output.values()), list(outputs[0].values()), equal_nan=True
        )
    return outputs[0]


@pytest.mark.parametrize("n_workers", [1, 3, 6])
def test_matches_reference_and_is_independent_of_worker_count(n_workers):
    y = np.array([1.0, 2.0, 4.0, 7.0])
    predictions = np.array([1.5, 2.5, 3.0, 6.0])
    actual = score(y, predictions, n_workers=n_workers)
    rss = np.square(y - predictions).sum()
    tss = np.square(y - y.mean()).sum()
    assert actual == pytest.approx(
        {
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "mae": mean_absolute_error(y, predictions),
            "r2": r2_score(y, predictions),
            "f_stat": (tss - rss) * 18 / rss,
        }
    )


@pytest.mark.parametrize("p,n_train", [(0, 20), (2, 3), (2, 2)])
def test_undefined_f_does_not_erase_prediction_metrics(p, n_train):
    actual = score([1, 2, 3], [2, 3, 4], p=p, n_train=n_train)
    assert actual["rmse"] == 1.0
    assert actual["mae"] == 1.0
    assert actual["r2"] == -0.5
    assert np.isnan(actual["f_stat"])


@pytest.mark.parametrize("prediction", [3.0, 4.0])
def test_constant_outcome_preserves_errors(prediction):
    actual = score([3, 3, 3], [prediction] * 3)
    assert actual["rmse"] == abs(3.0 - prediction)
    assert actual["mae"] == abs(3.0 - prediction)
    assert np.isnan(actual["r2"])
    assert np.isnan(actual["f_stat"])


def test_perfect_fit():
    actual = score([1, 2, 3], [1, 2, 3])
    assert actual == {"rmse": 0.0, "mae": 0.0, "r2": 1.0, "f_stat": np.inf}


def test_effectively_perfect_fit_does_not_round_errors_to_zero():
    y = np.array([1.0, 2.0, 3.0])
    actual = score(y, y + 1e-8)
    assert actual["rmse"] > 0
    assert actual["mae"] > 0
    assert actual["r2"] == pytest.approx(1.0)
    assert actual["f_stat"] == np.inf


def test_single_test_observation():
    actual = score([2], [3], n_workers=2)
    assert actual["rmse"] == 1
    assert actual["mae"] == 1
    assert np.isnan(actual["r2"])
    assert np.isnan(actual["f_stat"])


def test_empty_global_test_set_fails():
    with pytest.raises(ValueError, match="nonempty global test set"):
        score([], [], n_workers=2)


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
def test_outcome_scale_does_not_change_r_squared(scale):
    actual = score(np.array([1, 2, 3]) * scale, np.array([2, 3, 4]) * scale)
    assert actual["r2"] == pytest.approx(-0.5)
    assert actual["rmse"] == pytest.approx(scale)


def test_negative_r_squared_is_preserved():
    actual = score([1, 2, 3], [10, 10, 10])
    assert actual["r2"] == pytest.approx(r2_score([1, 2, 3], [10, 10, 10]))
    assert actual["f_stat"] < 0
