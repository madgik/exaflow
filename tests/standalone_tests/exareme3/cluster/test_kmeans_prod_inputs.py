"""Exercise production K-means payloads on their real CSV fixtures without HTTP."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.exareme3.cluster.kmeans import KMeans
from exaflow.algorithms.exareme3.cluster.kmeans import (
    build_kmeans_preprocessing_parameters,
)
from exaflow.algorithms.exareme3.cluster.kmeans_cluster_creator import (
    KMeansClusterCreator,
)
from exaflow.algorithms.exareme3.linear_model.logistic_regression import (
    local_step as logistic_step,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from tests.prod_env_tests import test_kmeans as prod_kmeans
from tests.prod_env_tests import test_kmeans_reusable_downstream as prod_downstream
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)

DATA_FOLDER = Path(__file__).resolve().parents[3] / "test_data" / "dementia_v_0_1"


def _load_data(payload, extra_variables=()):
    variables = list(
        dict.fromkeys(payload["inputdata"]["variables"] + list(extra_variables))
    )
    return pd.concat(
        [
            pd.read_csv(DATA_FOLDER / f"{dataset}.csv", usecols=variables)
            for dataset in payload["inputdata"]["datasets"]
        ],
        ignore_index=True,
    ).dropna(subset=variables)


def _run_report(payload, n_workers):
    data = _load_data(payload)
    coordinator = AggregationCoordinator(n_workers)

    def run(worker_id):
        try:
            algorithm = KMeans(
                engine=SimpleNamespace(),
                logger=Mock(),
                inputdata=Inputdata.model_validate(payload["inputdata"]),
                y=payload["algorithm"]["y"],
                parameters=payload["algorithm"]["parameters"],
            )
            client = SimulatedAggClient(worker_id=worker_id, coordinator=coordinator)
            algorithm.run_local_udf = lambda **kwargs: kwargs["func"](
                data=data.iloc[worker_id::n_workers],
                agg_client=client,
                **kwargs["kw_args"],
            )
            return algorithm.run()
        except Exception as exc:
            coordinator.abort(exc)
            raise

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run, worker_id) for worker_id in range(n_workers)]
        return [future.result(timeout=20) for future in futures]


@pytest.mark.parametrize("n_workers", [1, 3])
@pytest.mark.parametrize(
    "payload",
    prod_kmeans.get_test_inputs() + [prod_downstream._reporter_payload()],
    ids=["hippocampus", "amygdala", "saved_centers"],
)
def test_production_success_inputs_pass_privacy_and_export_replay(payload, n_workers):
    reports = _run_report(payload, n_workers)

    assert all(report == reports[0] for report in reports)
    result = reports[0]
    assert result.selected_k == payload["algorithm"]["parameters"]["k"]
    assert result.reusable_preprocessing.centers == {
        cluster.cluster_id: cluster.center for cluster in result.clusters
    }
    assert all(not cluster.size_interval.startswith("<") for cluster in result.clusters)


@pytest.mark.parametrize("n_workers", [1, 3])
@pytest.mark.parametrize("k", [3, 4])
def test_original_hippocampus_inputs_still_reject_the_singleton(k, n_workers):
    payload = prod_kmeans.get_test_inputs()[0]
    payload["algorithm"]["parameters"]["k"] = k

    with pytest.raises(BadInputError, match="minimum cohort size"):
        _run_report(payload, n_workers)


def test_production_saved_centers_still_feed_logistic_regression():
    payload = prod_downstream._reporter_payload()
    report = _run_report(payload, n_workers=1)[0]
    parameters = build_kmeans_preprocessing_parameters(
        report.reusable_preprocessing, new_column_names=[prod_downstream.GENERATED_CODE]
    )
    data = KMeansClusterCreator(params=parameters).transform_data(
        data=_load_data(payload, extra_variables=["gender"])
    )

    summary = logistic_step(
        agg_client=SimulatedAggClient(
            worker_id=0, coordinator=AggregationCoordinator(1)
        ),
        data=data,
        positive_class="F",
        y_var="gender",
        x_vars=[prod_downstream.GENERATED_CODE],
        categorical_vars=[prod_downstream.GENERATED_CODE],
        numerical_vars=[],
    )

    assert data[prod_downstream.GENERATED_CODE].nunique() == 3
    assert np.isfinite(summary["coefficients"]).all()
    assert any(
        prod_downstream.GENERATED_CODE in name for name in summary["feature_names"]
    )
