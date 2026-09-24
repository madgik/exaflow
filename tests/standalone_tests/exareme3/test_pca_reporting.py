import json

import numpy as np
import pandas as pd

from exaflow.algorithms.exareme3.decomposition.pca import local_step
from exaflow.algorithms.exareme3.decomposition.pca_common import PCAExecutionResult
from exaflow.algorithms.exareme3.decomposition.pca_common import PCAResult
from exaflow.algorithms.exareme3.decomposition.pca_common import (
    PCAReusablePreprocessing,
)
from exaflow.algorithms.exareme3.decomposition.pca_common import build_pca_result
from exaflow.algorithms.exareme3.decomposition.pca_common import run_federated_pca
from exaflow.algorithms.exareme3.decomposition.pca_with_transformations import (
    local_step as transformed_local_step,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    _simulate_federated_execution,
)

VARIABLES = ["age", "bmi", "crp"]
DATA = pd.DataFrame(
    [
        [1.0, 2.0, 4.0],
        [2.0, 1.0, 3.0],
        [3.0, 5.0, 2.0],
        [4.0, 4.0, 6.0],
        [5.0, 8.0, 5.0],
        [6.0, 7.0, 7.0],
        [7.0, 11.0, 8.0],
        [8.0, 10.0, 9.0],
    ],
    columns=VARIABLES,
)


def _run_local_step(step, data, **kwargs):
    parts = [part.copy() for part in np.array_split(data, 2)]

    def worker_fn(worker_id, agg_client):
        return step(
            agg_client=agg_client,
            data=parts[worker_id],
            y_vars=VARIABLES,
            **kwargs,
        )

    return _simulate_federated_execution(2, worker_fn)[0]


def _run_shared_execution(data, *, compute_scores, data_transformation=None):
    parts = [part.copy() for part in np.array_split(data, 1)]

    def worker_fn(worker_id, agg_client):
        return run_federated_pca(
            agg_client=agg_client,
            data=parts[worker_id],
            variables=VARIABLES,
            data_transformation=data_transformation,
            compute_scores=compute_scores,
        )

    return _simulate_federated_execution(1, worker_fn)[0]


def _plain_result():
    payload = _run_local_step(local_step, DATA)
    return build_pca_result(
        payload=payload,
        variables=VARIABLES,
        pca_variant="pca",
    )


def test_plain_pca_report_has_legacy_fields_and_component_report():
    result = _plain_result()

    assert result.title == "Eigenvalues and Eigenvectors"
    assert result.n_obs == len(DATA)
    assert len(result.eigenvalues) == len(VARIABLES)
    assert len(result.eigenvectors) == len(VARIABLES)
    assert [component.component_id for component in result.components] == [
        "PC1",
        "PC2",
        "PC3",
    ]
    assert result.variables == VARIABLES
    assert result.components[0].loadings.keys() == set(VARIABLES)
    assert list(result.components[0].loadings) == VARIABLES


def test_explained_variance_and_cumulative_values_are_consistent():
    result = _plain_result()
    ratios = [component.explained_variance_ratio for component in result.components]
    cumulative = [
        component.cumulative_explained_variance for component in result.components
    ]

    assert all(np.isfinite(ratio) for ratio in ratios)
    assert np.isclose(sum(ratios), 1.0)
    assert all(left <= right for left, right in zip(cumulative, cumulative[1:]))
    assert np.isclose(cumulative[-1], 1.0)
    np.testing.assert_allclose(
        [component.explained_variance for component in result.components],
        result.eigenvalues,
    )


def test_plain_recipe_is_typed_json_round_trippable_and_has_no_fitted_state():
    result = _plain_result()
    recipe = result.reusable_preprocessing
    dumped = recipe.model_dump()
    restored = PCAReusablePreprocessing.model_validate(json.loads(json.dumps(dumped)))

    assert restored == recipe
    assert restored.schema_version == "1"
    assert restored.preprocessing_name == "pca_column_creator"
    assert restored.base_parameters.variables == VARIABLES
    assert restored.base_parameters.pca_variant == "pca"
    assert restored.base_parameters.data_transformation is None
    assert [choice.component_id for choice in restored.component_choices] == [
        "PC1",
        "PC2",
        "PC3",
    ]
    forbidden = {"scores", "means", "scales", "covariance", "eigenvectors"}

    def assert_private_state_absent(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for nested in value.values():
                assert_private_state_absent(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_private_state_absent(nested)

    assert_private_state_absent(dumped)


def test_transformed_pca_report_preserves_variant_and_transformation_recipe():
    transformation = {"log": ["age"], "exp": ["crp"]}
    payload = _run_local_step(
        transformed_local_step,
        DATA,
        data_transformation=transformation,
    )
    result = build_pca_result(
        payload=payload,
        variables=VARIABLES,
        pca_variant="pca_with_transformation",
        data_transformation=transformation,
    )

    assert result.components[0].component_id == "PC1"
    assert (
        result.reusable_preprocessing.base_parameters.pca_variant
        == "pca_with_transformation"
    )
    assert (
        result.reusable_preprocessing.base_parameters.data_transformation
        == transformation
    )


def test_legacy_pca_result_shape_remains_constructible():
    result = PCAResult(
        title="Eigenvalues and Eigenvectors",
        n_obs=2,
        eigenvalues=[1.0],
        eigenvectors=[[1.0]],
    )

    assert result.n_obs == 2
    assert result.eigenvalues == [1.0]
    assert result.eigenvectors == [[1.0]]
    assert result.components == []


def test_shared_execution_without_scores_returns_only_the_fitted_model():
    execution = _run_shared_execution(DATA, compute_scores=False)

    assert isinstance(execution, PCAExecutionResult)
    assert execution.model.n_samples_seen_ == len(DATA)
    assert execution.scores is None
    assert not hasattr(execution, "transformed_data")


def test_shared_execution_with_scores_returns_all_component_scores():
    execution = _run_shared_execution(DATA, compute_scores=True)

    assert execution.scores is not None
    assert execution.scores.shape == (len(DATA), len(VARIABLES))
    assert np.all(np.isfinite(execution.scores))
    assert not hasattr(execution, "transformed_data")


def test_shared_execution_fits_once_and_projects_once_per_call(monkeypatch):
    from exaflow.algorithms.exareme3.decomposition import pca_common

    fit_calls = 0
    transform_calls = 0
    original_fit = pca_common.FederatedPCA.fit
    original_transform = pca_common.FederatedPCA.transform

    def fit_spy(self, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, *args, **kwargs)

    def transform_spy(self, *args, **kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return original_transform(self, *args, **kwargs)

    monkeypatch.setattr(pca_common.FederatedPCA, "fit", fit_spy)
    monkeypatch.setattr(pca_common.FederatedPCA, "transform", transform_spy)

    _run_shared_execution(DATA, compute_scores=False)
    assert fit_calls == 1
    assert transform_calls == 0

    _run_shared_execution(DATA, compute_scores=True)
    assert fit_calls == 2
    assert transform_calls == 1


def test_transformed_score_calculation_does_not_repeat_log_or_exp(monkeypatch):
    from exaflow.algorithms.exareme3.decomposition import pca_common

    log_calls = 0
    exp_calls = 0
    original_log = pca_common.np.log
    original_exp = pca_common.np.exp

    def log_spy(*args, **kwargs):
        nonlocal log_calls
        log_calls += 1
        return original_log(*args, **kwargs)

    def exp_spy(*args, **kwargs):
        nonlocal exp_calls
        exp_calls += 1
        return original_exp(*args, **kwargs)

    monkeypatch.setattr(pca_common.np, "log", log_spy)
    monkeypatch.setattr(pca_common.np, "exp", exp_spy)

    execution = _run_shared_execution(
        DATA,
        compute_scores=True,
        data_transformation={"log": ["age"], "exp": ["crp"]},
    )

    assert execution.scores is not None
    assert log_calls == 1
    assert exp_calls == 1


def test_score_calculation_does_not_mutate_source_dataframe():
    source = DATA.copy(deep=True)
    before = source.copy(deep=True)

    _run_shared_execution(
        source,
        compute_scores=True,
        data_transformation={"log": ["age"], "exp": ["crp"]},
    )

    pd.testing.assert_frame_equal(source, before)
