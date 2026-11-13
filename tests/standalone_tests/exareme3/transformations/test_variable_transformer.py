"""Nonlinear transformations use the shared fit and preserve source values."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from scipy.stats import yeojohnson

from exaflow.algorithms.exareme3.transformations.variable_transformer import (
    VariableTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.exareme3.transformations.step_test_support import metadata
from tests.standalone_tests.exareme3.transformations.step_test_support import (
    run_workers,
)
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


@pytest.mark.parametrize(
    "method,function", [("log1p", np.log1p), ("sqrt", np.sqrt), ("asinh", np.arcsinh)]
)
def test_elementwise_derived_output(method, function):
    source = pd.DataFrame({"crp": np.linspace(0, 10, 40)})
    results = run_workers(
        VariableTransformer,
        {"source_variable": "crp", "new_column_names": ["prepared"], "method": method},
        [source.iloc[:5], source.iloc[5:]],
        metadata(),
    )
    actual = pd.concat([item[1][0] for item in results])
    np.testing.assert_allclose(actual["prepared"], function(source["crp"]))
    pd.testing.assert_series_equal(actual["crp"], source["crp"])


def test_power_transform_and_replay_match_scipy():
    source = pd.DataFrame({"crp": [-4.0, -2, -1, 0, 1, 3, 8]})
    params = {
        "source_variable": "crp",
        "new_column_names": ["prepared"],
        "method": "yeo_johnson",
    }
    results = run_workers(
        VariableTransformer, params, [source.iloc[:2], source.iloc[2:]], metadata()
    )
    actual = pd.concat([item[1][0] for item in results])
    np.testing.assert_allclose(
        actual["prepared"], yeojohnson(source["crp"])[0], atol=0.001
    )
    step = VariableTransformer(
        params={**params, "recipe": asdict(results[0][0].fitted_recipe_)}
    )
    replayed, _ = step.transform_data_and_metadata(
        data=source.copy(), metadata=metadata()
    )
    np.testing.assert_array_equal(replayed["prepared"], actual["prepared"])


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_quantile_transform_replays_achieved_fit(distribution):
    source = pd.DataFrame({"crp": np.linspace(0, 10, 80)})
    params = {
        "source_variable": "crp",
        "new_column_names": ["prepared"],
        "method": "quantile_" + distribution,
        "n_quantiles": 5,
    }
    results = run_workers(
        VariableTransformer, params, [source.iloc[:3], source.iloc[3:]], metadata()
    )
    actual = pd.concat([item[1][0] for item in results])
    assert np.isfinite(actual["prepared"]).all()
    assert actual["prepared"].is_monotonic_increasing
    state = results[0][0].fitted_recipe_
    assert state.fit_diagnostics
    replay, _ = VariableTransformer(
        params={**params, "recipe": asdict(state)}
    ).transform_data_and_metadata(data=source.copy(), metadata=metadata())
    np.testing.assert_array_equal(actual["prepared"], replay["prepared"])


def test_quantile_fit_cannot_lower_the_platform_minimum(monkeypatch):
    from exaflow.worker import config

    monkeypatch.setattr(config.privacy, "minimum_row_count", 10)
    step = VariableTransformer(
        params={
            "source_variable": "crp",
            "new_column_names": ["prepared"],
            "method": "quantile_uniform",
            "n_quantiles": 5,
            "min_observations_per_interval": 1,
        }
    )
    with pytest.raises(BadInputError, match="global observations"):
        step.transform_data_and_metadata(
            data=pd.DataFrame({"crp": np.arange(39.0)}),
            metadata=metadata(),
            agg_client=DummyAggClient(),
        )
