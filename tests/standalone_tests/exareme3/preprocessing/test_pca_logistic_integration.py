import json

import numpy as np
import pandas as pd

from exaflow.algorithms.exareme3.decomposition.pca import local_step as pca_local_step
from exaflow.algorithms.exareme3.decomposition.pca_common import build_pca_result
from exaflow.algorithms.exareme3.linear_model.logistic_regression import (
    local_step as logistic_local_step,
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

VARIABLES = ["age", "bmi", "crp"]
OUTCOME = "outcome"


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
            OUTCOME: [
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
    return {
        "age": {
            "code": "age",
            "label": "age",
            "sql_type": "real",
            "is_categorical": False,
            "enumerations": None,
        },
        "bmi": {
            "code": "bmi",
            "label": "bmi",
            "sql_type": "real",
            "is_categorical": False,
            "enumerations": None,
        },
        "crp": {
            "code": "crp",
            "label": "crp",
            "sql_type": "real",
            "is_categorical": False,
            "enumerations": None,
        },
        OUTCOME: {
            "code": OUTCOME,
            "label": OUTCOME,
            "sql_type": "text",
            "is_categorical": True,
            "enumerations": {"yes": "yes", "no": "no"},
        },
    }


def _agg_client():
    coordinator = AggregationCoordinator(n_workers=1)
    return SimulatedAggClient(worker_id=0, coordinator=coordinator)


def _reporter_recipe():
    source_data = _data()[VARIABLES].copy()
    parts = [part.copy() for part in np.array_split(source_data, 2)]

    def worker_fn(worker_id, agg_client):
        return pca_local_step(
            agg_client=agg_client,
            data=parts[worker_id],
            y_vars=VARIABLES,
        )

    payload = _simulate_federated_execution(2, worker_fn)[0]
    report = build_pca_result(
        payload=payload,
        variables=VARIABLES,
        pca_variant="pca",
    )
    recipe_json = json.dumps(report.reusable_preprocessing.model_dump())
    return report, json.loads(recipe_json)


def _apply_pca_creator(*, selected_components):
    report, recipe = _reporter_recipe()
    data = _data()
    transformed, transformed_metadata = (
        udf_service._apply_preprocessing_steps_to_data_and_metadata(
            data=data,
            metadata=_metadata(),
            preprocessing=[
                {
                    "name": "pca_column_creator",
                    "parameters": {
                        "reusable_preprocessing": recipe,
                        "selected_components": selected_components,
                        "code_prefix": "pca",
                    },
                }
            ],
            check_min_rows=False,
            agg_client=_agg_client(),
        )
    )
    return report, transformed, transformed_metadata


def _run_logistic(*, data, metadata, predictors):
    return udf_service._execute_udf(
        udf=logistic_local_step,
        kw_args={
            "positive_class": "yes",
            "y_var": OUTCOME,
            "x_vars": predictors,
            "categorical_vars": [],
            "numerical_vars": predictors,
        },
        data=data,
        metadata=metadata,
        agg_client=_agg_client(),
        inject_agg_client=True,
    )


def test_pca_pc1_is_consumed_by_logistic_in_the_standalone_flow():
    source_before = _data()[VARIABLES].copy()
    report, transformed, transformed_metadata = _apply_pca_creator(
        selected_components=["PC1"]
    )

    assert report.reusable_preprocessing is not None
    assert "pca_PC1" in transformed
    assert transformed_metadata["pca_PC1"] == {
        "code": "pca_PC1",
        "label": "pca_PC1",
        "sql_type": "real",
        "is_categorical": False,
        "enumerations": None,
    }

    result = _run_logistic(
        data=transformed,
        metadata=transformed_metadata,
        predictors=["pca_PC1"],
    )

    assert result["n_obs"] == len(transformed)
    assert "feature_names" in result
    assert "pca_PC1" in result["feature_names"]
    assert np.isfinite(result["coefficients"]).all()
    pd.testing.assert_frame_equal(transformed[VARIABLES], source_before)


def test_pca_pc1_and_pc2_are_consumed_by_logistic_in_the_standalone_flow():
    _, transformed, transformed_metadata = _apply_pca_creator(
        selected_components=["PC1", "PC2"]
    )

    assert {"pca_PC1", "pca_PC2"}.issubset(transformed.columns)
    assert all(
        transformed_metadata[code]["sql_type"] == "real"
        and transformed_metadata[code]["is_categorical"] is False
        for code in ("pca_PC1", "pca_PC2")
    )

    result = _run_logistic(
        data=transformed,
        metadata=transformed_metadata,
        predictors=["pca_PC1", "pca_PC2"],
    )

    assert result["n_obs"] == len(transformed)
    assert {"pca_PC1", "pca_PC2"}.issubset(result["feature_names"])
    assert len(result["coefficients"]) == 3
