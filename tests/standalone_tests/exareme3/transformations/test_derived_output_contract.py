"""Existing specification, controller, worker and downstream paths are reused."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from exaflow import exareme3_preprocessing_step_classes
from exaflow.algorithms.exareme3.missing_values.global_imputer import GlobalImputer
from exaflow.algorithms.exareme3.missing_values.missing_indicator import (
    MissingIndicatorStep,
)
from exaflow.algorithms.exareme3.outliers.global_winsorizer import GlobalWinsorizer
from exaflow.algorithms.exareme3.outliers.outlier_detector import OutlierDetectorStep
from exaflow.algorithms.exareme3.transformations.variable_scaler import VariableScaler
from exaflow.algorithms.exareme3.transformations.variable_transformer import (
    VariableTransformer,
)
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.federated.decomposition import FederatedPCA
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.specifications import ComponentType
from exaflow.algorithms.utils.output_specifications import resolve_output_metadata
from exaflow.controller.services.api.algorithm_spec_dtos import (
    _get_preprocessing_step_specifications_dtos,
)
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.controller.services.api.analysis_request_validator import (
    _validate_and_apply_preprocessing,
)
from exaflow.controller.services.specifications import Specifications
from exaflow.worker.exareme3.udf.udf_service import (
    _apply_preprocessing_steps_to_data_and_metadata,
)
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement
from tests.standalone_tests.exareme3.transformations.step_test_support import metadata
from tests.standalone_tests.exareme3.transformations.step_test_support import (
    run_workers,
)
from tests.standalone_tests.exareme3.transformations.step_test_support import validate
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)

CASES = [
    (VariableScaler, {"method": "standard"}),
    (VariableTransformer, {"method": "log1p"}),
    (GlobalImputer, {"strategy": "mean"}),
    (MissingIndicatorStep, {}),
    (GlobalWinsorizer, {}),
    (OutlierDetectorStep, {"mode": "zscore"}),
]


@pytest.mark.parametrize("step_class,options", CASES)
def test_steps_use_the_existing_registration_and_output_contract(
    step_class, options, monkeypatch
):
    spec = step_class.get_specification()
    assert exareme3_preprocessing_step_classes[spec.name] is step_class
    assert PreprocessingStep in step_class.__bases__
    assert (
        step_class.transform_data_and_metadata
        is PreprocessingStep.transform_data_and_metadata
    )
    assert spec.outputs.new_column_names_param == "new_column_names"
    assert len(spec.outputs.columns) == 1
    params = {"source_variable": "crp", "new_column_names": ["derived"], **options}
    expected = resolve_output_metadata(spec, params, metadata())
    from exaflow.controller import config

    monkeypatch.setattr(config.aggregation_server, "enabled", True)
    enabled = Specifications().enabled_preprocessing_steps
    actual_dtos = json.loads(
        _get_preprocessing_step_specifications_dtos(
            list(enabled.values())
        ).model_dump_json()
    )
    dto = next(item for item in actual_dtos if item["name"] == spec.name)
    assert dto["outputs"]["new_column_names_param"] == "new_column_names"
    frame = pd.DataFrame({"crp": np.linspace(0, 8, 80)})
    step = step_class(params=params)
    validate(step, metadata())
    transformed, transformed_metadata = step.transform_data_and_metadata(
        data=frame.copy(), metadata=metadata(), agg_client=DummyAggClient()
    )
    pd.testing.assert_series_equal(transformed["crp"], frame["crp"])
    assert transformed_metadata["crp"] == metadata()["crp"]
    assert transformed_metadata["derived"] == expected["derived"]
    assert step.transform_variables(variables=["crp"]) == ["crp", "derived"]
    assert (ComponentType.AGGREGATION_SERVER in spec.components) == (
        step_class is not MissingIndicatorStep
    )


@pytest.mark.parametrize("names", [[], ["a", "b"], ["crp"], [" "], None])
@pytest.mark.parametrize("step_class,options", CASES)
def test_generated_name_mismatch_and_collisions_fail_before_execution(
    step_class, options, names
):
    with pytest.raises(BadUserInput):
        validate(
            step_class(
                params={"source_variable": "crp", "new_column_names": names, **options}
            ),
            metadata(),
        )


def test_named_recipe_rejects_wrong_source_method_and_corruption():
    params = {
        "source_variable": "crp",
        "new_column_names": ["derived"],
        "method": "standard",
    }
    step = VariableScaler(params=params)
    step.transform_data_and_metadata(
        data=pd.DataFrame({"crp": [1.0, 2.0]}),
        metadata=metadata(),
        agg_client=DummyAggClient(),
    )
    recipe = asdict(step.fitted_recipe_)
    invalid = [
        {**recipe, "feature_names": ["another"]},
        {**recipe, "primitive_name": "maxabs_scaler"},
        {**recipe, "scale": [0]},
        {**recipe, "location": [np.inf]},
    ]
    for state in invalid:
        with pytest.raises(BadUserInput):
            validate(VariableScaler(params={**params, "recipe": state}), metadata())


def test_controller_and_worker_support_a_derived_chain_and_downstream_pca():
    preprocessing = [
        {
            "name": "variable_transformer",
            "parameters": {
                "source_variable": "crp",
                "new_column_names": ["crp_log"],
                "method": "log1p",
            },
        },
        {
            "name": "variable_scaler",
            "parameters": {
                "source_variable": "crp_log",
                "new_column_names": ["crp_prepared"],
                "method": "minmax",
            },
        },
    ]
    request = AnalysisRequestDTO.model_validate(
        {
            "inputdata": {
                "data_model": "dm:0.1",
                "datasets": ["d1"],
                "variables": ["crp"],
            },
            "preprocessing": preprocessing,
            "algorithm": {"name": "pca", "y": ["crp_prepared"]},
        }
    )
    source_cde = CommonDataElement(
        code="crp", label="CRP", sql_type="real", is_categorical=False
    )
    inputdata, cdes = _validate_and_apply_preprocessing(
        analysis_request_dto=request,
        preprocessing_steps_specs={
            name: cls.get_specification()
            for name, cls in exareme3_preprocessing_step_classes.items()
        },
        data_model_cdes={"crp": source_cde},
    )
    assert inputdata.variables == ["crp", "crp_log", "crp_prepared"]
    assert cdes["crp_prepared"].sql_type == "real"
    source = pd.DataFrame({"crp": np.linspace(0, 20, 80)})
    frame, final_metadata = _apply_preprocessing_steps_to_data_and_metadata(
        data=source.copy(),
        metadata=metadata(),
        preprocessing=preprocessing,
        check_min_rows=True,
        agg_client=DummyAggClient(),
    )
    pd.testing.assert_series_equal(frame["crp"], source["crp"])
    expected = np.log1p(source["crp"])
    np.testing.assert_allclose(frame["crp_prepared"], expected / expected.max())
    assert final_metadata["crp_prepared"]["sql_type"] == "real"
    model = FederatedPCA(DummyAggClient(), copy=True)
    model.fit(frame[["crp_prepared"]].to_numpy())
    assert model.components_.shape == (1, 1)


def test_missing_indicator_works_without_aggregation_and_preserves_missing_source():
    source = pd.DataFrame({"crp": [1.0, np.nan, 3]})
    result, _ = MissingIndicatorStep(
        params={"source_variable": "crp", "new_column_names": ["missing"]}
    ).transform_data_and_metadata(data=source.copy(), metadata=metadata())
    np.testing.assert_array_equal(result["missing"], [0, 1, 0])
    assert result["missing"].dtype.kind in "iu"
    pd.testing.assert_series_equal(result["crp"], source["crp"])


@pytest.mark.parametrize(
    "strategy,expected", [("mean", 3), ("median", 3), ("constant", 0)]
)
def test_imputation_is_global_and_only_changes_derived_values(strategy, expected):
    source = pd.DataFrame({"crp": [1.0, 3.0, 5.0, np.nan]})
    results = run_workers(
        GlobalImputer,
        {
            "source_variable": "crp",
            "new_column_names": ["filled"],
            "strategy": strategy,
        },
        [source.iloc[:2], source.iloc[2:]],
        metadata(),
    )
    actual = pd.concat([item[1][0] for item in results])
    pd.testing.assert_series_equal(actual["crp"], source["crp"])
    np.testing.assert_allclose(actual["filled"], [1, 3, 5, expected])


def test_outlier_flag_is_diagnostic_only_and_integer_typed():
    source = pd.DataFrame({"crp": [0.0, 0, 0, 10, np.nan]})
    result, _ = OutlierDetectorStep(
        params={
            "source_variable": "crp",
            "new_column_names": ["outlier"],
            "mode": "zscore",
            "threshold": 1.5,
        }
    ).transform_data_and_metadata(
        data=source.copy(), metadata=metadata(), agg_client=DummyAggClient()
    )
    pd.testing.assert_series_equal(result["crp"], source["crp"])
    np.testing.assert_array_equal(result["outlier"], [0, 0, 0, 1, 0])
    assert result["outlier"].dtype.kind in "iu"


def test_winsorization_only_clips_derived_values_and_replays_bounds():
    source = pd.DataFrame({"crp": [0.0, 1, 2, 3, 4, 100]})
    params = {
        "source_variable": "crp",
        "new_column_names": ["clipped"],
        "lower_quantile": 0.0,
        "upper_quantile": 0.8,
    }
    results = run_workers(
        GlobalWinsorizer, params, [source.iloc[:2], source.iloc[2:]], metadata()
    )
    actual = pd.concat([item[1][0] for item in results])
    recipe = results[0][0].fitted_recipe_
    expected = np.clip(source["crp"], recipe.lower_bounds[0], recipe.upper_bounds[0])
    np.testing.assert_allclose(actual["clipped"], expected)
    assert actual["clipped"].iloc[-1] < source["crp"].iloc[-1]
    pd.testing.assert_series_equal(actual["crp"], source["crp"])


@pytest.mark.parametrize(
    "step_class,options",
    [
        (VariableTransformer, {"method": [], "lambda_tolerance": 0}),
        (VariableTransformer, {"method": "yeo_johnson", "lambda_tolerance": 0}),
        (VariableTransformer, {"method": "quantile_uniform", "n_quantiles": 2.5}),
        (GlobalImputer, {"strategy": "most_frequent"}),
        (GlobalWinsorizer, {"lower_quantile": 0.9, "upper_quantile": 0.1}),
        (OutlierDetectorStep, {"mode": "zscore", "threshold": 0}),
    ],
)
def test_invalid_method_settings_fail_during_request_validation(step_class, options):
    with pytest.raises(BadUserInput):
        validate(
            step_class(
                params={
                    "source_variable": "crp",
                    "new_column_names": ["derived"],
                    **options,
                }
            ),
            metadata(),
        )


def test_aggregation_component_gating_is_not_bypassed(monkeypatch):
    from exaflow.controller import config

    monkeypatch.setattr(config.aggregation_server, "enabled", False)
    enabled = Specifications().enabled_preprocessing_steps
    for step_class, _ in CASES:
        assert (step_class.get_specification().name in enabled) == (
            step_class is MissingIndicatorStep
        )


def test_numerical_steps_reject_categorical_source_before_fitting():
    categorical = metadata(sql_type="text", categorical=True)
    for step_class, options in CASES:
        if step_class is MissingIndicatorStep:
            continue
        with pytest.raises(BadUserInput, match="numerical"):
            validate(
                step_class(
                    params={
                        "source_variable": "crp",
                        "new_column_names": ["derived"],
                        **options,
                    }
                ),
                categorical,
            )


def test_missing_indicator_supports_categorical_values_without_sharing_them():
    source = pd.DataFrame({"crp": ["a", None, "b"]})
    step = MissingIndicatorStep(
        params={"source_variable": "crp", "new_column_names": ["missing"]}
    )
    validate(step, metadata(sql_type="text", categorical=True))
    result, _ = step.transform_data_and_metadata(
        data=source.copy(), metadata=metadata(sql_type="text", categorical=True)
    )
    np.testing.assert_array_equal(result["missing"], [0, 1, 0])
    pd.testing.assert_series_equal(result["crp"], source["crp"])


def test_worker_double_transport_preserves_replay_and_integer_settings():
    from google.protobuf.json_format import MessageToDict
    from google.protobuf.json_format import ParseDict
    from google.protobuf.struct_pb2 import Struct

    source = pd.DataFrame({"crp": np.linspace(0, 8, 80)})
    params = {
        "source_variable": "crp",
        "new_column_names": ["prepared"],
        "method": "quantile_uniform",
        "n_quantiles": 5,
        "n_bins": 16,
        "max_refinement_rounds": 1,
    }
    wire_params = MessageToDict(ParseDict(params, Struct()))
    step = VariableTransformer(params=wire_params)
    fitted, _ = step.transform_data_and_metadata(
        data=source.copy(), metadata=metadata(), agg_client=DummyAggClient()
    )
    replay_params = MessageToDict(
        ParseDict({**params, "recipe": asdict(step.fitted_recipe_)}, Struct())
    )
    replayed, _ = VariableTransformer(params=replay_params).transform_data_and_metadata(
        data=source.copy(), metadata=metadata()
    )
    np.testing.assert_array_equal(fitted["prepared"], replayed["prepared"])


def test_public_recipe_shape_roundtrips_through_request_dto():
    source = pd.DataFrame({"crp": [1.0, 2.0, 4.0]})
    params = {
        "source_variable": "crp",
        "new_column_names": ["prepared"],
        "method": "standard",
    }
    step = VariableScaler(params=params)
    step.transform_data_and_metadata(
        data=source.copy(), metadata=metadata(), agg_client=DummyAggClient()
    )
    payload = {
        "inputdata": {"data_model": "dm:0.1", "datasets": ["d1"], "variables": ["crp"]},
        "preprocessing": [
            {
                "name": "variable_scaler",
                "parameters": {**params, "recipe": asdict(step.fitted_recipe_)},
            }
        ],
        "algorithm": {"name": "pca", "y": ["prepared"]},
    }
    request = AnalysisRequestDTO.model_validate_json(json.dumps(payload))
    assert request.model_dump()["preprocessing"][0]["parameters"]["recipe"] == asdict(
        step.fitted_recipe_
    )


def test_runtime_rejects_a_dataframe_output_collision_before_aggregation():
    source = pd.DataFrame({"crp": [1.0, 2.0], "hidden": [9, 8]})
    step = VariableScaler(
        params={
            "source_variable": "crp",
            "new_column_names": ["hidden"],
            "method": "standard",
        }
    )
    with pytest.raises(BadInputError, match="derived column must be new"):
        step.transform_data_and_metadata(
            data=source.copy(), metadata=metadata(), agg_client=DummyAggClient()
        )
