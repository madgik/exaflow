import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from exaflow.algorithms.exareme3.model_selection.cross_validation.linear_regression_cv import (
    LinearRegressionCV,
)
from exaflow.algorithms.exareme3.model_selection.cross_validation.linear_regression_cv import (
    local_step,
)
from exaflow.algorithms.federated import FederatedOLS
from exaflow.algorithms.federated.model_selection import FederatedCrossValidator
from exaflow.algorithms.federated.model_selection import FederatedKFoldSplitter
from exaflow.algorithms.federated.model_selection import FederatedRegressionScorer
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    _simulate_federated_execution,
)
from tests.testcase_generators.linear_regression_cv_testcase_generator import (
    LinearRegressionTestCaseGenerator,
)


@pytest.mark.parametrize("n_workers", [1, 3])
@pytest.mark.parametrize("categorical", [False, True])
def test_cv_matches_reference_with_identical_local_folds(n_workers, categorical):
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {"x": rng.normal(size=91), "group": np.repeat([0, 1, 2], [30, 30, 31])}
    )
    data["y"] = 2 + 3 * data.x + data.group + rng.normal(size=len(data))
    parts = [data.iloc[idx] for idx in np.array_split(np.arange(len(data)), n_workers)]

    def worker(worker_id, agg_client):
        part = parts[worker_id]
        if categorical:
            return local_step(
                agg_client, part, "y", ["group", "x"], ["group"], ["x"], 5
            )
        return FederatedCrossValidator(
            estimator=FederatedOLS(),
            splitter=FederatedKFoldSplitter(5),
            scorer=FederatedRegressionScorer(),
        ).evaluate(part[["x"]].to_numpy(), part.y.to_numpy(), agg_client=agg_client)

    outputs = _simulate_federated_execution(n_workers, worker)
    expected = {key: [] for key in ["rmse", "mae", "r2", "f_stat", "n_obs"]}
    local_folds = [list(KFold(5).split(part)) for part in parts]
    for fold in range(5):
        train = pd.concat(
            [part.iloc[splits[fold][0]] for part, splits in zip(parts, local_folds)]
        )
        test = pd.concat(
            [part.iloc[splits[fold][1]] for part, splits in zip(parts, local_folds)]
        )

        def design(frame):
            if categorical:
                return np.column_stack(
                    [frame.group == 1, frame.group == 2, frame.x]
                ).astype(float)
            return frame[["x"]].to_numpy()

        X_train, X_test = design(train), design(test)
        prediction = LinearRegression().fit(X_train, train.y).predict(X_test)
        rss = np.square(test.y - prediction).sum()
        tss = np.square(test.y - test.y.mean()).sum()
        p = X_test.shape[1]
        expected["rmse"].append(np.sqrt(mean_squared_error(test.y, prediction)))
        expected["mae"].append(mean_absolute_error(test.y, prediction))
        expected["r2"].append(r2_score(test.y, prediction))
        expected["f_stat"].append((tss - rss) * (len(train) - p - 1) / (p * rss))
        expected["n_obs"].append(len(train))

    for output in outputs:
        for key, values in expected.items():
            np.testing.assert_allclose(output[key], values, rtol=1e-8, atol=1e-10)
        if categorical:
            assert output["feature_names"] == ["Intercept", "group[1]", "group[2]", "x"]

    if n_workers == 1:
        # Exercise reference output generation without loading the fixture database.
        generator = object.__new__(LinearRegressionTestCaseGenerator)
        reference = generator.compute_expected_output(
            (data[["y"]], data[["group", "x"]] if categorical else data[["x"]]),
            {"n_splits": 5},
            [{"code": "group", "isCategorical": True}],
        )
        for metric, field in [
            ("rmse", "root_mean_sq_error"),
            ("mae", "mean_abs_error"),
            ("r2", "r_squared"),
            ("f_stat", "f_stat"),
        ]:
            np.testing.assert_allclose(
                reference[field],
                [np.mean(outputs[0][metric]), np.std(outputs[0][metric], ddof=1)],
                rtol=1e-8,
                atol=1e-10,
            )
        assert reference["n_obs"] == outputs[0]["n_obs"]
        assert "mean_sq_error" not in reference


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("metric,field", [("r2", "r_squared"), ("f_stat", "f_stat")])
def test_response_renames_rmse_and_preserves_other_metric_summaries(
    nonfinite, metric, field
):
    metrics = {
        "feature_names": ["Intercept", "x"],
        "n_obs": [8, 8, 8, 8, 8],
        "rmse": [1, 2, 3, 4, 5],
        "mae": [1, 1, 2, 2, 3],
        "r2": [-1, 0, 0.5, 0.7, 0.9],
        "f_stat": [2, 3, 4, 5, 6],
    }
    metrics[metric][2] = nonfinite
    algorithm = LinearRegressionCV(
        engine=Mock(run_udf=Mock(return_value=[metrics, metrics])),
        logger=Mock(),
        inputdata=Inputdata(
            data_model="test:1", datasets=["test"], variables=["x", "y"]
        ),
        x=["x"],
        y=["y"],
        metadata={"x": {"is_categorical": False}},
        parameters={"n_splits": 5},
    )
    result = json.loads(algorithm.run().model_dump_json())
    assert "mean_sq_error" not in result
    assert result["root_mean_sq_error"] == pytest.approx(
        [3, np.std([1, 2, 3, 4, 5], ddof=1)]
    )
    assert result["mean_abs_error"] == pytest.approx(
        [1.8, np.std([1, 1, 2, 2, 3], ddof=1)]
    )
    assert result[field] == [None, None]
    assert result["n_obs"] == [8] * 5


def test_constant_categorical_predictor_keeps_intercept_only_error_metrics():
    data = pd.DataFrame({"group": [1] * 30, "y": np.arange(30, dtype=float)})

    def worker(worker_id, agg_client):
        return local_step(agg_client, data, "y", ["group"], ["group"], [], 5)

    metrics = _simulate_federated_execution(1, worker)[0]
    assert metrics["feature_names"] == ["Intercept"]
    assert np.isnan(metrics["f_stat"]).all()
    for fold, (train, test) in enumerate(KFold(5).split(data)):
        predicted = np.full(len(test), data.y.iloc[train].mean())
        assert metrics["rmse"][fold] == pytest.approx(
            np.sqrt(mean_squared_error(data.y.iloc[test], predicted))
        )
        assert metrics["mae"][fold] == pytest.approx(
            mean_absolute_error(data.y.iloc[test], predicted)
        )
        assert metrics["r2"][fold] == pytest.approx(
            r2_score(data.y.iloc[test], predicted)
        )
