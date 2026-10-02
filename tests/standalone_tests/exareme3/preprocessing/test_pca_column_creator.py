import json

import numpy as np
import pandas as pd
import pytest

from exaflow import exareme3_preprocessing_step_classes
from exaflow.algorithms.exareme3.decomposition import pca_common
from exaflow.algorithms.exareme3.decomposition.pca import local_step as pca_local_step
from exaflow.algorithms.exareme3.decomposition.pca_common import PCAComponentChoice
from exaflow.algorithms.exareme3.decomposition.pca_common import (
    PCAReusableBaseParameters,
)
from exaflow.algorithms.exareme3.decomposition.pca_common import (
    PCAReusablePreprocessing,
)
from exaflow.algorithms.exareme3.decomposition.pca_common import build_pca_result
from exaflow.algorithms.exareme3.decomposition.pca_common import run_federated_pca
from exaflow.algorithms.exareme3.decomposition.pca_with_transformations import (
    local_step as transformed_pca_local_step,
)
from exaflow.algorithms.exareme3.preprocessing import pca_column_creator
from exaflow.algorithms.exareme3.preprocessing.pca_column_creator import (
    PCAColumnCreator,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    _simulate_federated_execution,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)

VARIABLES = ["age", "bmi", "crp"]
PCA_COLUMN_CREATOR_NAME = "pca_column_creator"


def _recipe(*, variant="pca", data_transformation=None):
    return PCAReusablePreprocessing(
        base_parameters=PCAReusableBaseParameters(
            variables=VARIABLES,
            pca_variant=variant,
            data_transformation=data_transformation,
        ),
        component_choices=[
            PCAComponentChoice(component_id="PC1", explained_variance_ratio=0.6),
            PCAComponentChoice(component_id="PC2", explained_variance_ratio=0.3),
            PCAComponentChoice(component_id="PC3", explained_variance_ratio=0.1),
        ],
    )


def _inputdata():
    return Inputdata(
        data_model="dm:0.1",
        datasets=["dataset"],
        variables=VARIABLES,
    )


def _metadata():
    return {
        variable: {"is_categorical": False, "sql_type": "real"}
        for variable in VARIABLES
    }


def _data():
    return pd.DataFrame(
        {
            "age": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "bmi": [2.0, 1.0, 5.0, 4.0, 8.0, 7.0],
            "crp": [4.0, 3.0, 2.0, 6.0, 5.0, 7.0],
        },
        index=[10, 11, 12, 20, 21, 22],
    )


def _agg_client():
    coordinator = AggregationCoordinator(n_workers=1)
    return SimulatedAggClient(worker_id=0, coordinator=coordinator)


def _reporter_result(*, variant, data_transformation=None):
    step = pca_local_step if variant == "pca" else transformed_pca_local_step
    parts = [part.copy() for part in np.array_split(_data(), 2)]

    def worker_fn(worker_id, agg_client):
        kwargs = {
            "agg_client": agg_client,
            "data": parts[worker_id],
            "y_vars": VARIABLES,
        }
        if variant == "pca_with_transformation":
            kwargs["data_transformation"] = data_transformation
        return step(**kwargs)

    payload = _simulate_federated_execution(2, worker_fn)[0]
    return build_pca_result(
        payload=payload,
        variables=VARIABLES,
        pca_variant=variant,
        data_transformation=data_transformation,
    )


def _creator_from_reporter(*, result, selected_components):
    recipe_json = json.dumps(result.reusable_preprocessing.model_dump())
    recipe_dict = json.loads(recipe_json)
    creator = PCAColumnCreator(
        params={
            "reusable_preprocessing": recipe_dict,
            "selected_components": selected_components,
            "code_prefix": "pca",
        }
    )
    return _validate(creator)


def _creator(*, selected_components=None, code_prefix="pca", recipe=None):
    return PCAColumnCreator(
        params={
            "reusable_preprocessing": (recipe or _recipe()).model_dump(),
            "selected_components": (
                ["PC1"] if selected_components is None else selected_components
            ),
            "code_prefix": code_prefix,
        }
    )


def _validate(creator):
    creator.validate_params(inputdata=_inputdata(), metadata=_metadata())
    return creator


def test_valid_plain_and_transformed_recipes_parse():
    _validate(_creator(recipe=_recipe()))
    _validate(
        _creator(
            recipe=_recipe(
                variant="pca_with_transformation",
                data_transformation={"log": ["age"]},
            )
        )
    )


def test_creator_uses_recipe_for_pca_configuration_only():
    spec = PCAColumnCreator.get_specification()

    assert set(spec.parameters) == {
        "reusable_preprocessing",
        "selected_components",
        "code_prefix",
    }
    assert "pca_variant" not in spec.parameters
    assert "data_transformation" not in spec.parameters
    assert "variables" not in spec.parameters


@pytest.mark.parametrize(
    "recipe, message",
    [
        (_recipe(variant="pca", data_transformation={"log": ["age"]}), "must not"),
        (
            _recipe(variant="pca_with_transformation"),
            "must include",
        ),
    ],
)
def test_recipe_variant_requires_consistent_transformation(recipe, message):
    with pytest.raises(BadUserInput, match=message):
        _validate(_creator(recipe=recipe))


def test_recipe_identity_and_schema_version_are_validated():
    recipe = _recipe().model_dump()
    recipe["preprocessing_name"] = "other_creator"
    with pytest.raises(BadUserInput, match="preprocessing_name"):
        _validate(_creator(recipe=PCAReusablePreprocessing.model_validate(recipe)))

    recipe = _recipe().model_dump()
    recipe["schema_version"] = "2"
    with pytest.raises(BadUserInput, match="schema_version"):
        _validate(_creator(recipe=PCAReusablePreprocessing.model_validate(recipe)))


@pytest.mark.parametrize("selected", [["PC1", "PC2"], ["PC2", "PC1"]])
def test_selected_components_are_accepted_in_caller_order(selected):
    creator = _validate(_creator(selected_components=selected))
    assert creator.transform_variables(variables=VARIABLES)[-len(selected) :] == [
        f"pca_{component}" for component in selected
    ]


@pytest.mark.parametrize(
    "selected, message",
    [
        ([], "at least one"),
        (["PC99"], "not available"),
        (["PC1", "PC1"], "duplicates"),
        (["pc1"], "not available"),
        ([1], "strings"),
    ],
)
def test_selected_components_are_strictly_validated(selected, message):
    with pytest.raises(BadUserInput, match=message):
        _validate(_creator(selected_components=selected))


def test_default_and_custom_prefixes_generate_ordered_codes():
    default_creator = _validate(_creator(selected_components=["PC1"]))
    assert default_creator.transform_variables(variables=VARIABLES) == [
        *VARIABLES,
        "pca_PC1",
    ]

    custom_creator = _validate(
        _creator(selected_components=["PC2", "PC1"], code_prefix="clinical_pca")
    )
    assert custom_creator.transform_variables(variables=VARIABLES)[-2:] == [
        "clinical_pca_PC2",
        "clinical_pca_PC1",
    ]


@pytest.mark.parametrize("prefix", ["", "   "])
def test_blank_prefix_is_rejected(prefix):
    with pytest.raises(BadUserInput, match="code_prefix"):
        _validate(_creator(code_prefix=prefix))


def test_source_variables_remain_and_recipe_variables_are_required():
    creator = _validate(_creator(selected_components=["PC1", "PC2"]))
    assert creator.transform_variables(variables=VARIABLES) == [
        *VARIABLES,
        "pca_PC1",
        "pca_PC2",
    ]

    recipe = _recipe().model_dump()
    recipe["base_parameters"]["variables"] = ["missing"]
    with pytest.raises(BadUserInput, match="inputdata.variables"):
        _validate(_creator(recipe=PCAReusablePreprocessing.model_validate(recipe)))


def test_transform_metadata_creates_real_non_categorical_outputs():
    creator = _validate(_creator(selected_components=["PC2", "PC1"]))
    transformed = creator.transform_metadata(metadata=_metadata())

    assert transformed["pca_PC2"] == {
        "code": "pca_PC2",
        "label": "pca_PC2",
        "sql_type": "real",
        "is_categorical": False,
        "enumerations": None,
    }
    assert transformed["pca_PC1"]["is_categorical"] is False


def test_specification_declares_multiple_numerical_output_and_aggregation():
    spec = PCAColumnCreator.get_specification()

    assert spec.output.type.value == "new_numerical_column"
    assert spec.output.multiple is True
    assert spec.components[0].value == "AGGREGATION_SERVER"


def test_creator_is_discoverable_through_normal_registration():
    assert (
        exareme3_preprocessing_step_classes[PCA_COLUMN_CREATOR_NAME] is PCAColumnCreator
    )


def test_transform_data_does_not_fabricate_checkpoint_5a_scores():
    creator = _validate(_creator())
    transformed = creator.transform_data(data=_data(), agg_client=_agg_client())
    assert np.isfinite(transformed["pca_PC1"]).all()


@pytest.mark.parametrize("selected", [["PC1"], ["PC2"], ["PC1", "PC2"]])
def test_plain_creator_generates_selected_score_columns(selected):
    creator = _validate(_creator(selected_components=selected))
    transformed = creator.transform_data(data=_data(), agg_client=_agg_client())

    assert list(transformed.columns) == [*VARIABLES, *[f"pca_{pc}" for pc in selected]]
    assert all(np.isfinite(transformed[f"pca_{pc}"]).all() for pc in selected)


def test_requested_component_order_maps_to_correct_scores():
    data = _data()
    creator = _validate(_creator(selected_components=["PC2", "PC1"]))
    transformed = creator.transform_data(data=data, agg_client=_agg_client())

    expected = run_federated_pca(
        agg_client=_agg_client(),
        data=_data(),
        variables=VARIABLES,
        data_transformation=None,
        compute_scores=True,
    ).scores
    np.testing.assert_allclose(transformed["pca_PC2"], expected[:, 1])
    np.testing.assert_allclose(transformed["pca_PC1"], expected[:, 0])
    assert list(transformed.columns[-2:]) == ["pca_PC2", "pca_PC1"]


def test_creator_preserves_source_columns_and_values():
    data = _data()
    source_before = data[VARIABLES].copy()
    creator = _validate(_creator(selected_components=["PC1", "PC2"]))

    transformed = creator.transform_data(data=data, agg_client=_agg_client())

    pd.testing.assert_frame_equal(transformed[VARIABLES], source_before)


def test_creator_uses_one_shared_fit_and_projection_for_multiple_components(
    monkeypatch,
):
    helper_calls = 0
    fit_calls = 0
    transform_calls = 0
    original_helper = pca_column_creator.run_federated_pca
    original_fit = pca_common.FederatedPCA.fit
    original_transform = pca_common.FederatedPCA.transform

    def helper_spy(*args, **kwargs):
        nonlocal helper_calls
        helper_calls += 1
        return original_helper(*args, **kwargs)

    def fit_spy(self, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, *args, **kwargs)

    def transform_spy(self, *args, **kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return original_transform(self, *args, **kwargs)

    monkeypatch.setattr(pca_column_creator, "run_federated_pca", helper_spy)
    monkeypatch.setattr(pca_common.FederatedPCA, "fit", fit_spy)
    monkeypatch.setattr(pca_common.FederatedPCA, "transform", transform_spy)

    creator = _validate(_creator(selected_components=["PC1", "PC2"]))
    creator.transform_data(data=_data(), agg_client=_agg_client())

    assert helper_calls == 1
    assert fit_calls == 1
    assert transform_calls == 1


@pytest.mark.parametrize(
    "data_transformation",
    [
        {"log": ["age"]},
        {"exp": ["age"]},
        {"center": ["age"]},
        {"standardize": ["age"]},
        {
            "log": ["age"],
            "exp": ["crp"],
            "center": ["bmi"],
            "standardize": ["crp"],
        },
    ],
)
def test_transformed_creator_scores_match_shared_execution(data_transformation):
    creator = _validate(
        _creator(
            selected_components=["PC1"],
            recipe=_recipe(
                variant="pca_with_transformation",
                data_transformation=data_transformation,
            ),
        )
    )
    transformed = creator.transform_data(data=_data(), agg_client=_agg_client())
    expected = run_federated_pca(
        agg_client=_agg_client(),
        data=_data(),
        variables=VARIABLES,
        data_transformation=data_transformation,
        compute_scores=True,
    ).scores

    np.testing.assert_allclose(transformed["pca_PC1"], expected[:, 0])


def test_transformed_creator_preserves_requested_order_and_multiple_outputs():
    data_transformation = {"log": ["age"], "standardize": ["bmi", "crp"]}
    creator = _validate(
        _creator(
            selected_components=["PC2", "PC1"],
            recipe=_recipe(
                variant="pca_with_transformation",
                data_transformation=data_transformation,
            ),
        )
    )
    transformed = creator.transform_data(data=_data(), agg_client=_agg_client())
    expected = run_federated_pca(
        agg_client=_agg_client(),
        data=_data(),
        variables=VARIABLES,
        data_transformation=data_transformation,
        compute_scores=True,
    ).scores

    assert list(transformed.columns[-2:]) == ["pca_PC2", "pca_PC1"]
    np.testing.assert_allclose(transformed["pca_PC2"], expected[:, 1])
    np.testing.assert_allclose(transformed["pca_PC1"], expected[:, 0])


@pytest.mark.parametrize(
    "selected_components", [["PC1"], ["PC1", "PC2"], ["PC2", "PC1"]]
)
def test_reporter_plain_recipe_json_replays_through_creator(selected_components):
    report = _reporter_result(variant="pca")
    recipe = report.reusable_preprocessing
    data = _data()
    source_before = data[VARIABLES].copy()
    creator = _creator_from_reporter(
        result=report,
        selected_components=selected_components,
    )

    transformed = creator.transform_data(data=data, agg_client=_agg_client())
    expected = run_federated_pca(
        agg_client=_agg_client(),
        data=_data(),
        variables=recipe.base_parameters.variables,
        data_transformation=recipe.base_parameters.data_transformation,
        compute_scores=True,
    ).scores

    assert recipe.schema_version == "1"
    assert recipe.preprocessing_name == "pca_column_creator"
    assert recipe.base_parameters.variables == VARIABLES
    assert recipe.base_parameters.pca_variant == "pca"
    assert recipe.base_parameters.data_transformation is None
    assert [choice.component_id for choice in recipe.component_choices] == [
        component.component_id for component in report.components
    ]
    assert [choice.explained_variance_ratio for choice in recipe.component_choices] == [
        component.explained_variance_ratio for component in report.components
    ]
    assert list(transformed.columns[-len(selected_components) :]) == [
        f"pca_{component}" for component in selected_components
    ]
    for component in selected_components:
        component_index = int(component[2:]) - 1
        np.testing.assert_allclose(
            transformed[f"pca_{component}"], expected[:, component_index]
        )
    pd.testing.assert_frame_equal(transformed[VARIABLES], source_before)


@pytest.mark.parametrize(
    "data_transformation, selected_components",
    [
        ({"log": ["age"]}, ["PC1"]),
        (
            {"log": ["age"], "standardize": ["bmi", "crp"]},
            ["PC1", "PC2"],
        ),
    ],
)
def test_reporter_transformed_recipe_json_replays_through_creator(
    data_transformation, selected_components
):
    report = _reporter_result(
        variant="pca_with_transformation",
        data_transformation=data_transformation,
    )
    recipe = report.reusable_preprocessing
    data = _data()
    source_before = data[VARIABLES].copy()
    creator = _creator_from_reporter(
        result=report,
        selected_components=selected_components,
    )

    transformed = creator.transform_data(data=data, agg_client=_agg_client())
    expected = run_federated_pca(
        agg_client=_agg_client(),
        data=_data(),
        variables=recipe.base_parameters.variables,
        data_transformation=recipe.base_parameters.data_transformation,
        compute_scores=True,
    ).scores

    assert recipe.base_parameters.pca_variant == "pca_with_transformation"
    assert recipe.base_parameters.data_transformation == data_transformation
    assert list(transformed.columns[-len(selected_components) :]) == [
        f"pca_{component}" for component in selected_components
    ]
    for component in selected_components:
        component_index = int(component[2:]) - 1
        np.testing.assert_allclose(
            transformed[f"pca_{component}"], expected[:, component_index]
        )
    pd.testing.assert_frame_equal(transformed[VARIABLES], source_before)


def test_reporter_recipe_replay_keeps_one_fit_and_one_projection(monkeypatch):
    report = _reporter_result(
        variant="pca_with_transformation", data_transformation={"log": ["age"]}
    )
    helper_calls = 0
    fit_calls = 0
    transform_calls = 0
    original_helper = pca_column_creator.run_federated_pca
    original_fit = pca_common.FederatedPCA.fit
    original_transform = pca_common.FederatedPCA.transform

    def helper_spy(*args, **kwargs):
        nonlocal helper_calls
        helper_calls += 1
        return original_helper(*args, **kwargs)

    def fit_spy(self, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, *args, **kwargs)

    def transform_spy(self, *args, **kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return original_transform(self, *args, **kwargs)

    monkeypatch.setattr(pca_column_creator, "run_federated_pca", helper_spy)
    monkeypatch.setattr(pca_common.FederatedPCA, "fit", fit_spy)
    monkeypatch.setattr(pca_common.FederatedPCA, "transform", transform_spy)

    creator = _creator_from_reporter(
        result=report,
        selected_components=["PC1", "PC2"],
    )
    creator.transform_data(data=_data(), agg_client=_agg_client())

    assert helper_calls == 1
    assert fit_calls == 1
    assert transform_calls == 1
