import json

import numpy as np
import pandas as pd

from exaflow.algorithms.exareme3.decomposition.pca import PCAResult
from exaflow.algorithms.exareme3.decomposition.pca import PCAReusablePreprocessing
from exaflow.algorithms.exareme3.decomposition.pca import build_pca_result
from exaflow.algorithms.exareme3.decomposition.pca import local_step
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


def _run_local_step(step, data):
    parts = [part.copy() for part in np.array_split(data, 2)]

    def worker_fn(worker_id, agg_client):
        return step(
            agg_client=agg_client,
            data=parts[worker_id],
            y_vars=VARIABLES,
        )

    return _simulate_federated_execution(2, worker_fn)[0]


def _plain_result():
    payload = _run_local_step(local_step, DATA)
    return build_pca_result(payload=payload, variables=VARIABLES)


def test_plain_pca_report_contains_all_components_and_cumulative_variance():
    result = _plain_result()

    assert result.title == "Eigenvalues and Eigenvectors"
    assert result.n_obs == len(DATA)
    assert [component.component_id for component in result.components] == [
        "PC1",
        "PC2",
        "PC3",
    ]
    ratios = [component.explained_variance_ratio for component in result.components]
    cumulative = [
        component.cumulative_explained_variance for component in result.components
    ]
    assert np.isclose(sum(ratios), 1.0)
    assert all(left <= right for left, right in zip(cumulative, cumulative[1:]))
    assert np.isclose(cumulative[-1], 1.0)


def test_plain_recipe_is_typed_json_round_trippable_without_fitted_state():
    result = _plain_result()
    recipe = result.reusable_preprocessing
    restored = PCAReusablePreprocessing.model_validate(
        json.loads(json.dumps(recipe.model_dump()))
    )

    assert restored == recipe
    assert restored.base_parameters.variables == VARIABLES
    assert restored.base_parameters.pca_variant == "pca"
    assert [choice.component_id for choice in restored.component_choices] == [
        "PC1",
        "PC2",
        "PC3",
    ]


def test_legacy_pca_result_shape_remains_constructible():
    result = PCAResult(
        title="Eigenvalues and Eigenvectors",
        n_obs=2,
        eigenvalues=[1.0],
        eigenvectors=[[1.0]],
    )
    assert result.n_obs == 2
