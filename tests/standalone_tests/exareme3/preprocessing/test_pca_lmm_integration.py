import json

import numpy as np
import pandas as pd

from exaflow.algorithms.exareme3.decomposition.pca import local_step as pca_local_step
from exaflow.algorithms.exareme3.decomposition.pca_common import build_pca_result
from exaflow.algorithms.exareme3.mixed_effects.lmm import lmm_local_step
from exaflow.worker.exareme3.udf import udf_service
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    _simulate_federated_execution,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)

PCA_VARIABLES = ["age", "bmi", "crp"]
RESPONSE = "response"
GROUP = "group"


def _data():
    groups = np.repeat([f"group_{index}" for index in range(8)], 8)
    within_group = np.tile(np.arange(8, dtype=float), 8)
    age = 20.0 + within_group + np.repeat(np.arange(8, dtype=float) * 0.7, 8)
    bmi = (
        22.0 + (within_group % 4) * 0.8 + np.repeat(np.arange(8, dtype=float) * 0.2, 8)
    )
    crp = 1.0 + (within_group % 5) * 0.5 + np.repeat(np.arange(8, dtype=float) * 0.1, 8)
    group_effect = np.repeat(np.linspace(-1.5, 1.5, 8), 8)
    response = 5.0 + 0.4 * age + 0.7 * bmi - 0.3 * crp + group_effect
    return pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "crp": crp,
            RESPONSE: response,
            GROUP: groups,
        },
        index=list(range(100, 164)),
    )


def _metadata():
    metadata = {
        variable: {
            "code": variable,
            "label": variable,
            "sql_type": "real",
            "is_categorical": False,
            "enumerations": None,
        }
        for variable in PCA_VARIABLES + [RESPONSE]
    }
    metadata[GROUP] = {
        "code": GROUP,
        "label": GROUP,
        "sql_type": "text",
        "is_categorical": True,
        "enumerations": {f"group_{index}": f"group_{index}" for index in range(8)},
    }
    return metadata


def _agg_client():
    coordinator = AggregationCoordinator(n_workers=1)
    return SimulatedAggClient(worker_id=0, coordinator=coordinator)


def _reporter_recipe():
    source_data = _data()[PCA_VARIABLES].copy()
    parts = [part.copy() for part in np.array_split(source_data, 2)]

    def worker_fn(worker_id, agg_client):
        return pca_local_step(
            agg_client=agg_client,
            data=parts[worker_id],
            y_vars=PCA_VARIABLES,
        )

    payload = _simulate_federated_execution(2, worker_fn)[0]
    report = build_pca_result(
        payload=payload,
        variables=PCA_VARIABLES,
        pca_variant="pca",
    )
    recipe_json = json.dumps(report.reusable_preprocessing.model_dump())
    return json.loads(recipe_json)


def test_pca_pc1_is_consumed_by_lmm_as_a_fixed_effect():
    data = _data()
    source_before = data.copy(deep=True)
    recipe = _reporter_recipe()
    preprocessing = [
        {
            "name": "missing_values_handler",
            "parameters": {
                "strategies": {
                    variable: "drop" for variable in PCA_VARIABLES + [RESPONSE, GROUP]
                }
            },
        },
        {
            "name": "pca_column_creator",
            "parameters": {
                "reusable_preprocessing": recipe,
                "selected_components": ["PC1"],
                "code_prefix": "pca",
            },
        },
    ]

    transformed, transformed_metadata = (
        udf_service._apply_preprocessing_steps_to_data_and_metadata(
            data=data,
            metadata=_metadata(),
            preprocessing=preprocessing,
            check_min_rows=False,
            agg_client=_agg_client(),
        )
    )

    assert set(transformed.columns) == {
        *PCA_VARIABLES,
        RESPONSE,
        GROUP,
        "pca_PC1",
    }
    assert transformed_metadata["pca_PC1"] == {
        "code": "pca_PC1",
        "label": "pca_PC1",
        "sql_type": "real",
        "is_categorical": False,
        "enumerations": None,
    }

    result = udf_service._execute_udf(
        udf=lmm_local_step,
        kw_args={
            "y_var": RESPONSE,
            "grouping_var": [GROUP],
            "categorical_vars": [],
            "numerical_vars": ["pca_PC1"],
        },
        data=transformed,
        metadata=transformed_metadata,
        agg_client=_agg_client(),
        inject_agg_client=True,
    )

    assert result["n_obs"] == len(transformed)
    assert result["n_groups"] == 8
    assert result["feature_names"] == ["Intercept", "pca_PC1"]
    assert len(result["coefficients"]) == len(result["feature_names"])
    assert len(result["std_err"]) == len(result["feature_names"])
    assert isinstance(result["converged"], bool)
    assert np.isfinite(result["coefficients"]).all()
    pd.testing.assert_frame_equal(transformed[source_before.columns], source_before)
