import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.decomposition.pca import PCAComponentChoice
from exaflow.algorithms.exareme3.decomposition.pca import PCAReusableBaseParameters
from exaflow.algorithms.exareme3.decomposition.pca import PCAReusablePreprocessing
from exaflow.algorithms.exareme3.decomposition.pca_column_creator import (
    PCAColumnCreator,
)
from exaflow.algorithms.federated.decomposition.pca import FederatedPCA
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.output_specifications import resolve_output_metadata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)

VARIABLES = ["age", "bmi", "crp"]


def _recipe():
    return PCAReusablePreprocessing(
        base_parameters=PCAReusableBaseParameters(variables=VARIABLES),
        component_choices=[
            PCAComponentChoice(component_id="PC1", explained_variance_ratio=0.6),
            PCAComponentChoice(component_id="PC2", explained_variance_ratio=0.3),
            PCAComponentChoice(component_id="PC3", explained_variance_ratio=0.1),
        ],
    ).model_dump()


def _inputdata():
    return Inputdata(data_model="dm:0.1", datasets=["dataset"], variables=VARIABLES)


def _metadata():
    return {variable: {"type": "continuous"} for variable in VARIABLES}


def _data():
    return pd.DataFrame(
        {
            "age": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "bmi": [2.0, 1.0, 5.0, 4.0, 8.0, 7.0],
            "crp": [4.0, 3.0, 2.0, 6.0, 5.0, 7.0],
        }
    )


def _creator(selected_components, new_column_names=None, n_components=None):
    if new_column_names is None:
        new_column_names = [
            f"pca_{component.lower()}" for component in selected_components
        ]
    if n_components is None:
        n_components = len(selected_components)
    return PCAColumnCreator(
        params={
            "reusable_preprocessing": _recipe(),
            "selected_components": selected_components,
            "new_column_names": new_column_names,
            "n_components": n_components,
        }
    )


def _independent_creator(selected_components, new_column_names=None, n_components=None):
    if new_column_names is None:
        new_column_names = [
            f"pca_{component.lower()}" for component in selected_components
        ]
    if n_components is None:
        n_components = len(selected_components)
    return PCAColumnCreator(
        params={
            "variables": VARIABLES,
            "selected_components": selected_components,
            "new_column_names": new_column_names,
            "n_components": n_components,
        }
    )


def _validate(creator):
    resolve_output_metadata(creator.get_specification(), creator._params, _metadata())
    creator.validate_params(inputdata=_inputdata(), metadata=_metadata())
    return creator


def _agg_client():
    coordinator = AggregationCoordinator(n_workers=1)
    return SimulatedAggClient(worker_id=0, coordinator=coordinator)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_creator_supports_one_two_and_more_dynamic_components(count):
    selected = [f"PC{index}" for index in range(1, count + 1)]
    names = [f"score_{index}" for index in range(1, count + 1)]
    creator = _validate(_creator(selected, names, count))

    transformed = creator.transform_data_and_metadata(
        data=_data(), metadata=_metadata(), agg_client=_agg_client()
    )
    data, metadata = transformed

    assert list(data.columns[-count:]) == names
    assert list(metadata)[-count:] == names
    assert all(metadata[name]["type"] == "continuous" for name in names)
    assert np.isfinite(data[names].to_numpy()).all()


def test_creator_scores_match_a_single_pca_fit_and_transform():
    data = pd.DataFrame(
        {
            "age": [101.0, 103.0, 107.0, 109.0, 113.0, 127.0],
            "bmi": [21.0, 31.0, 41.0, 51.0, 61.0, 71.0],
            "crp": [0.2, 0.8, 1.7, 2.9, 4.1, 5.6],
        }
    )
    creator = _validate(_creator(["PC1", "PC2"], ["score_1", "score_2"], 2))

    transformed, _ = creator.transform_data_and_metadata(
        data=data.copy(), metadata=_metadata(), agg_client=_agg_client()
    )

    pd.testing.assert_frame_equal(transformed[VARIABLES], data)

    reference_input = data.loc[:, VARIABLES].to_numpy(dtype=float, copy=True)
    reference_model = FederatedPCA(agg_client=_agg_client(), copy=True)
    reference_model.fit(reference_input)
    expected_scores = reference_model.transform(reference_input)

    np.testing.assert_allclose(
        transformed[["score_1", "score_2"]].to_numpy(),
        expected_scores[:, :2],
        rtol=1e-12,
        atol=1e-12,
    )


def test_creator_resolves_reordered_recipe_components_by_identifier():
    recipe = PCAReusablePreprocessing(
        base_parameters=PCAReusableBaseParameters(variables=VARIABLES),
        component_choices=[
            PCAComponentChoice(component_id="PC2", explained_variance_ratio=0.3),
            PCAComponentChoice(component_id="PC1", explained_variance_ratio=0.6),
            PCAComponentChoice(component_id="PC3", explained_variance_ratio=0.1),
        ],
    ).model_dump()
    creator = PCAColumnCreator(
        params={
            "reusable_preprocessing": recipe,
            "selected_components": ["PC2", "PC1"],
            "new_column_names": ["score_2", "score_1"],
            "n_components": 2,
        }
    )
    _validate(creator)
    data = _data()

    transformed, _ = creator.transform_data_and_metadata(
        data=data.copy(), metadata=_metadata(), agg_client=_agg_client()
    )

    reference_input = data.loc[:, VARIABLES].to_numpy(dtype=float, copy=True)
    reference_model = FederatedPCA(agg_client=_agg_client(), copy=True)
    reference_model.fit(reference_input)
    expected_scores = reference_model.transform(reference_input)

    np.testing.assert_allclose(transformed["score_2"], expected_scores[:, 1])
    np.testing.assert_allclose(transformed["score_1"], expected_scores[:, 0])


def test_creator_can_fit_pca_without_a_reusable_recipe():
    creator = _independent_creator(["PC1", "PC2"], ["score_1", "score_2"], 2)
    creator.validate_params(inputdata=_inputdata(), metadata=_metadata())

    transformed, metadata = creator.transform_data_and_metadata(
        data=_data(), metadata=_metadata(), agg_client=_agg_client()
    )

    assert list(transformed.columns[-2:]) == ["score_1", "score_2"]
    assert list(metadata)[-2:] == ["score_1", "score_2"]
    assert np.isfinite(transformed[["score_1", "score_2"]].to_numpy()).all()


@pytest.mark.parametrize(
    "params, message",
    [
        ({"n_components": 1, "new_column_names": ["one"]}, "n_components"),
        ({"new_column_names": ["one"]}, "new_column_names"),
        ({"selected_components": ["PC1", "PC4"]}, "not available"),
    ],
)
def test_creator_rejects_count_name_and_component_mismatches(params, message):
    selected = params.pop("selected_components", ["PC1", "PC2"])
    names = params.pop("new_column_names", ["one", "two"])
    count = params.pop("n_components", 2)
    creator = _creator(selected, names, count)
    with pytest.raises(BadUserInput, match=message):
        _validate(creator)


def test_creator_declares_dynamic_numerical_output_contract():
    specification = PCAColumnCreator.get_specification()

    assert specification.outputs.new_column_names_param == "new_column_names"
    assert len(specification.outputs.columns) == 1
    assert specification.outputs.columns[0].column_count_param == "n_components"
    assert specification.outputs.columns[0].type == "continuous"
    assert specification.parameters["reusable_preprocessing"].required is False
    assert specification.parameters["variables"].multiple is True
