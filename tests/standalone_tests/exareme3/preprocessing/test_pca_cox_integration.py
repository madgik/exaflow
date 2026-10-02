import json

import numpy as np
import pandas as pd

from exaflow.algorithms.exareme3.decomposition.pca import local_step as pca_local_step
from exaflow.algorithms.exareme3.decomposition.pca_common import build_pca_result
from exaflow.algorithms.exareme3.linear_model.cox_regression_classical import (
    cox_regression_classical_local_step,
)
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
TIME_VARIABLE = "survival_time"
EVENT_VARIABLE = "event_status"


def _data():
    return pd.DataFrame(
        {
            "age": np.arange(1.0, 21.0),
            "bmi": [
                2.0,
                5.0,
                1.0,
                4.0,
                8.0,
                7.0,
                3.0,
                6.0,
                9.0,
                2.5,
                5.5,
                1.5,
                4.5,
                8.5,
                7.5,
                3.5,
                6.5,
                9.5,
                2.2,
                5.2,
            ],
            "crp": [
                4.0,
                3.0,
                2.0,
                6.0,
                5.0,
                7.0,
                8.0,
                9.0,
                1.0,
                11.0,
                10.0,
                12.0,
                3.5,
                4.5,
                6.5,
                7.5,
                8.5,
                2.5,
                9.5,
                1.5,
            ],
            TIME_VARIABLE: np.arange(1.0, 21.0),
            EVENT_VARIABLE: [
                "no",
                "yes",
                "yes",
                "no",
                "yes",
                "no",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "no",
                "yes",
                "yes",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "yes",
            ],
        },
        index=list(range(100, 120)),
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
        for variable in PCA_VARIABLES + [TIME_VARIABLE]
    }
    metadata[EVENT_VARIABLE] = {
        "code": EVENT_VARIABLE,
        "label": EVENT_VARIABLE,
        "sql_type": "text",
        "is_categorical": True,
        "enumerations": {"yes": "yes", "no": "no"},
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


def test_pca_pc1_is_consumed_by_classical_cox():
    data = _data()
    source_before = data.copy(deep=True)
    recipe = _reporter_recipe()

    preprocessing = [
        {
            "name": "missing_values_handler",
            "parameters": {
                "strategies": {
                    variable: "drop"
                    for variable in PCA_VARIABLES + [TIME_VARIABLE, EVENT_VARIABLE]
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

    assert "pca_PC1" in transformed
    assert transformed_metadata["pca_PC1"] == {
        "code": "pca_PC1",
        "label": "pca_PC1",
        "sql_type": "real",
        "is_categorical": False,
        "enumerations": None,
    }
    assert {TIME_VARIABLE, EVENT_VARIABLE, *PCA_VARIABLES}.issubset(transformed.columns)

    result = udf_service._execute_udf(
        udf=cox_regression_classical_local_step,
        kw_args={
            "time_var": TIME_VARIABLE,
            "event_var": EVENT_VARIABLE,
            "positive_class": "yes",
            "categorical_vars": [],
            "numerical_vars": ["pca_PC1"],
        },
        data=transformed,
        metadata=transformed_metadata,
        agg_client=_agg_client(),
        inject_agg_client=True,
    )

    assert result["n_obs"] == len(transformed)
    assert "coefficients" in result
    assert "indep_vars" in result
    assert "pca_PC1" in result["indep_vars"]
    assert np.isfinite(result["coefficients"]).all()
    pd.testing.assert_frame_equal(transformed[source_before.columns], source_before)
