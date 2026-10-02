import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from exaflow.algorithms.exareme3.model_selection.cross_validation.logistic_regression_cv import (
    local_step,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    _simulate_federated_execution,
)


@pytest.mark.parametrize("n_workers", [1, 3])
def test_classification_cv_still_matches_identical_reference_folds(n_workers):
    rng = np.random.default_rng(7)
    x = rng.normal(size=150)
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-x))).astype(int)
    data = pd.DataFrame({"x": x, "y": y})
    parts = [data.iloc[idx] for idx in np.array_split(np.arange(len(data)), n_workers)]

    def worker(worker_id, agg_client):
        return local_step(agg_client, parts[worker_id], "y", 1, ["x"], [], ["x"], 5)

    outputs = _simulate_federated_execution(n_workers, worker)
    folds = [list(KFold(5).split(part)) for part in parts]
    for fold in range(5):
        train = pd.concat(
            [part.iloc[indices[fold][0]] for part, indices in zip(parts, folds)]
        )
        test = pd.concat(
            [part.iloc[indices[fold][1]] for part, indices in zip(parts, folds)]
        )
        model = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000)
        prediction = model.fit(train[["x"]], train.y).predict(test[["x"]])
        tn, fp, fn, tp = confusion_matrix(test.y, prediction, labels=[0, 1]).ravel()
        for output in outputs:
            assert [output[key][fold] for key in ["tn", "fp", "fn", "tp"]] == [
                tn,
                fp,
                fn,
                tp,
            ]
            assert output["n_obs"][fold] == len(train)
