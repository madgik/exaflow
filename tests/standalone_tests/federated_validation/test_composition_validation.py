from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.compose.passthrough import FederatedPassthrough
from exaflow.algorithms.federated.feature_engineering.one_hot_encoder import (
    FederatedOneHotEncoder,
)
from exaflow.algorithms.federated.feature_engineering.ordinal_encoder import (
    FederatedOrdinalEncoder,
)
from exaflow.algorithms.federated.model_selection.cross_validation.cross_validator import (
    FederatedCrossValidator,
)
from exaflow.algorithms.federated.model_selection.cross_validation.scorer_classification import (
    FederatedClassificationScorer,
)
from exaflow.algorithms.federated.model_selection.cross_validation.scorer_multiclass import (
    FederatedMulticlassClassificationScorer,
)
from exaflow.algorithms.federated.model_selection.cross_validation.scorer_regression import (
    FederatedRegressionScorer,
)
from exaflow.algorithms.federated.model_selection.cross_validation.splitter_kfold import (
    FederatedKFoldSplitter,
)
from exaflow.algorithms.federated.outliers.outlier_report import OutlierRule
from exaflow.algorithms.federated.outliers.outlier_report import WinsorizationHelper
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.algorithms.federated.utils import BadInputError


@pytest.mark.parametrize(
    "encoder", [FederatedOneHotEncoder, FederatedOrdinalEncoder, FederatedPassthrough]
)
def test_transformers_report_invalid_frame_types(encoder):
    with pytest.raises(BadInputError, match="DataFrame"):
        encoder().transform(None, categorical_vars=[], numerical_vars=[])


def test_ordinal_unknown_policy_is_checked_even_for_empty_columns():
    client = Mock()
    encoder = FederatedOrdinalEncoder(
        categories={"x": ["a"]}, handle_unknown="private-value"
    )
    with pytest.raises(BadInputError, match="handle_unknown"):
        encoder.fit(agg_client=client, data=pd.DataFrame(), categorical_vars=[])
    assert client.mock_calls == []


@pytest.mark.parametrize("transformer", [FederatedOneHotEncoder, FederatedPassthrough])
def test_numerical_conversion_errors_do_not_echo_data(transformer):
    with pytest.raises(BadInputError) as error:
        transformer().transform(
            pd.DataFrame({"x": ["private-value"]}),
            categorical_vars=[],
            numerical_vars=["x"],
        )
    assert "private-value" not in str(error.value)


def test_column_transformer_checks_options_before_fitting_components():
    component = Mock()
    client = Mock()
    transformer = FederatedColumnTransformer(
        [("x", component, ["x"])], remainder="private-value"
    )
    with pytest.raises(BadInputError, match="remainder"):
        transformer.fit(
            agg_client=client,
            data=pd.DataFrame({"x": [1]}),
            categorical_vars=[],
            numerical_vars=["x"],
        )
    component.fit.assert_not_called()
    assert client.mock_calls == []


def test_column_transformer_rejects_inconsistent_component_rows():
    component = Mock()
    component.transform.return_value = np.ones((1, 1))
    transformer = FederatedColumnTransformer([("x", component, ["x"])])
    transformer.fit(
        agg_client=Mock(),
        data=pd.DataFrame({"x": [1, 2]}),
        categorical_vars=[],
        numerical_vars=["x"],
    )
    with pytest.raises(BadInputError):
        transformer.transform(
            pd.DataFrame({"x": [1, 2]}), categorical_vars=[], numerical_vars=["x"]
        )


def test_pipeline_checks_outcome_rows_before_fitting_components():
    client = Mock()
    pipeline = FederatedPipeline(
        [("numeric", FederatedPassthrough()), ("model", Mock())]
    )
    with pytest.raises(BadInputError):
        pipeline.fit(
            agg_client=client,
            data=pd.DataFrame({"x": [1, 2]}),
            y=np.array([1]),
            categorical_vars=[],
            numerical_vars=["x"],
        )
    assert client.mock_calls == []


@pytest.mark.parametrize("n_splits", [1, 0, -1])
def test_kfold_checks_effective_split_count(n_splits):
    splitter = FederatedKFoldSplitter(n_splits)
    with pytest.raises(BadInputError, match="n_splits"):
        splitter.split_indices(5)


def test_kfold_checks_row_alignment_without_converting_buffers():
    splitter = FederatedKFoldSplitter(2)
    with pytest.raises(BadInputError):
        splitter.split(np.ones((4, 2)), np.ones(3))


def test_cross_validator_checks_frame_outcome_alignment_before_splitting():
    splitter = Mock()
    cv = FederatedCrossValidator(estimator=Mock(), splitter=splitter, scorer=Mock())
    with pytest.raises(BadInputError):
        cv.evaluate(
            None,
            np.ones(1),
            data=pd.DataFrame({"x": [1, 2]}),
            categorical_vars=[],
            numerical_vars=["x"],
            agg_client=Mock(),
        )
    splitter.split_indices.assert_not_called()


@pytest.mark.parametrize(
    "scorer",
    [
        FederatedRegressionScorer,
        FederatedClassificationScorer,
        lambda: FederatedMulticlassClassificationScorer(labels=[0, 1]),
    ],
)
def test_scorers_reject_broadcastable_prediction_lengths(scorer):
    results = Mock()
    results.predict.return_value = np.array([0.5])
    with pytest.raises(BadInputError):
        scorer().local(results, np.ones((2, 1)), np.array([0, 1]))


def test_binary_scorer_rejects_nonbinary_targets_before_prediction():
    results = Mock()
    with pytest.raises(BadInputError, match="encoded"):
        FederatedClassificationScorer().local(
            results, np.ones((2, 1)), np.array([0, 2])
        )
    results.predict.assert_not_called()


@pytest.mark.parametrize("thresholds", [[np.inf, 0.5, -np.inf], [0.8, 0.2], []])
def test_threshold_validation_keeps_existing_order_and_infinite_sentinels(thresholds):
    scorer = FederatedClassificationScorer(thresholds=thresholds)
    np.testing.assert_array_equal(scorer.thresholds, thresholds)


@pytest.mark.parametrize(
    "strategy, tail, fold",
    [
        ("private-value", "both", 1),
        ("iqr", "invalid", 1),
        ("quantile", "both", 0.5),
        ("iqr", "both", np.inf),
    ],
)
def test_direct_outlier_rules_are_validated(strategy, tail, fold):
    with pytest.raises(BadInputError):
        WinsorizationHelper.compute_bounds(
            pd.Series([1, 2]), OutlierRule("x", strategy, tail, fold)
        )


@pytest.mark.parametrize("categories", [["a", "a"], [None, "a"], None])
def test_ordinal_category_order_is_checked_without_aggregation(categories):
    client = Mock()
    encoder = FederatedOrdinalEncoder(categories={"x": categories})
    with pytest.raises(BadInputError, match="Category ordering"):
        encoder.fit(
            agg_client=client, data=pd.DataFrame({"x": ["a"]}), categorical_vars=["x"]
        )
    assert client.mock_calls == []


def test_column_transformer_retains_one_dimensional_component_outputs():
    component = Mock()
    component.transform.return_value = np.ones(2)
    transformer = FederatedColumnTransformer([("x", component, ["x"])])
    data = pd.DataFrame({"x": [1, 2]})
    transformer.fit(
        agg_client=Mock(), data=data, categorical_vars=[], numerical_vars=["x"]
    )
    assert transformer.transform(
        data, categorical_vars=[], numerical_vars=["x"]
    ).shape == (2, 1)


@pytest.mark.parametrize("n_train, p", [(-1, 1), (5, -1), (2.5, 1)])
def test_regression_scorer_validates_counts_before_aggregation(n_train, p):
    client = Mock()
    with pytest.raises(BadInputError):
        FederatedRegressionScorer().aggregate({}, client, n_train, p)
    assert client.mock_calls == []


def test_kfold_random_state_errors_are_reported_before_splitting():
    with pytest.raises(BadInputError, match="K-fold"):
        FederatedKFoldSplitter(
            2, shuffle=True, random_state="private-value"
        ).split_indices(4)
