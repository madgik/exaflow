"""Global fitting, source preservation and exact scaler replay."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MaxAbsScaler as ReferenceMaxAbs
from sklearn.preprocessing import MinMaxScaler as ReferenceMinMax
from sklearn.preprocessing import StandardScaler as ReferenceStandard

from exaflow.algorithms.exareme3.transformations.variable_scaler import VariableScaler
from tests.standalone_tests.exareme3.transformations.step_test_support import metadata
from tests.standalone_tests.exareme3.transformations.step_test_support import (
    run_workers,
)


@pytest.mark.parametrize(
    "method,reference",
    [
        ("standard", ReferenceStandard),
        ("minmax", ReferenceMinMax),
        ("maxabs", ReferenceMaxAbs),
    ],
)
def test_scaling_matches_reference_without_overwriting_source(method, reference):
    pooled = pd.DataFrame({"crp": np.linspace(2, 20, 40)})
    params = {
        "source_variable": "crp",
        "new_column_names": ["scaled"],
        "method": method,
    }
    results = run_workers(
        VariableScaler, params, [pooled.iloc[:7], pooled.iloc[7:]], metadata()
    )
    output = pd.concat([result[1][0] for result in results])
    np.testing.assert_allclose(
        output["scaled"], reference().fit_transform(pooled)[..., 0]
    )
    pd.testing.assert_series_equal(output["crp"], pooled["crp"])
    assert results[0][1][1]["crp"] == metadata()["crp"]
    assert results[0][1][1]["scaled"]["sql_type"] == "real"
    assert "min" not in results[0][1][1]["scaled"]
    replay = VariableScaler(
        params={**params, "recipe": asdict(results[0][0].fitted_recipe_)}
    )
    replayed, _ = replay.transform_data_and_metadata(
        data=pooled.copy(), metadata=metadata()
    )
    np.testing.assert_array_equal(replayed["scaled"], output["scaled"])


def test_robust_scaling_uses_pooled_quartiles():
    pooled = pd.DataFrame({"crp": np.linspace(0, 10, 1001)})
    results = run_workers(
        VariableScaler,
        {"source_variable": "crp", "new_column_names": ["scaled"], "method": "robust"},
        [pooled.iloc[:10], pooled.iloc[10:]],
        metadata(),
    )
    output = pd.concat([item[1][0] for item in results])
    np.testing.assert_allclose(output["scaled"], (pooled["crp"] - 5) / 5, atol=0.02)
