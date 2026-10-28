"""
End-to-end lazy aggregation checks for every aggregation-server algorithm.

Each algorithm runs in-process through the real worker path
(`udf_service.run_udf`, DuckDB loading, preprocessing, gRPC-shaped argument and
result round-trips) against the real `AggregationServer` servicer, with three
simulated workers holding different datasets. Every algorithm runs twice, once
with lazy aggregation disabled and once enabled, and the test checks that:

- results are identical,
- lazy aggregation needs no more server round-trips than `MAX_LAZY_ROUNDS`.
"""

import copy
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from exaflow import exareme3_algorithm_classes
from exaflow import exareme3_preprocessing_step_classes
from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.aggregation_server import config as aggregation_config
from exaflow.aggregation_server.server import AggregationServer
from exaflow.algorithms.exareme3.utils.metadata_enums import add_ordered_enums
from exaflow.algorithms.exareme3.utils.registry import get_udf_registry_key
from exaflow.algorithms.specifications import ComponentType
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from exaflow.udf_result_serialization import decode_udf_result
from exaflow.udf_result_serialization import encode_udf_result
from exaflow.worker import config as worker_config
from exaflow.worker.exareme3.udf import udf_service
from exaflow.worker.grpc_server import _dict_to_struct
from exaflow.worker.grpc_server import _struct_to_dict
from exaflow.worker.utils.duck_db_csv_loader import load_all_csvs_from_data_folder
from exaflow.worker.worker_info import duckdb_adapter
from exaflow.worker.worker_info.worker_info_db import get_data_model_cdes
from exaflow.worker_communication import InsufficientDataError
from exaflow.worker_communication import RunUdfSystemArgs

TEST_DATA_PATH = Path(__file__).parents[2] / "test_data"
N_WORKERS = 3
DATASETS = [
    "desd-synthdata0",
    "desd-synthdata1",
    "edsd0",
    "edsd1",
    "ppmi0",
    "ppmi1",
]
LONGITUDINAL_DATASETS = [
    "longitudinal_dementia0",
    "longitudinal_dementia1",
    "longitudinal_dementia2",
]
LONGITUDINAL_PREPROCESSING = [
    {
        "name": "longitudinal_transformer",
        "parameters": {
            "visit1": "BL",
            "visit2": "FL2",
            "strategies": {
                "timesincebaseline": "second",
                "alzheimerbroadcategory": "second",
                "lefthippocampus": "first",
                "leftamygdala": "first",
            },
        },
    }
]


def _case(x, y, parameters=None, *, datasets=None, **extra):
    return {
        "datasets": datasets or DATASETS,
        "x": x,
        "y": y,
        "parameters": parameters or {},
        **extra,
    }


CASES = {
    "anova_oneway": _case(["alzheimerbroadcategory"], ["lefthippocampus"]),
    "anova_twoway": _case(["agegroup", "gender"], ["lefthippocampus"], {"sstype": 2}),
    "binned_mann_whitney_u_test": _case(
        ["dataset"],
        ["lefthippocampus"],
        {
            "groupA": "edsd0",
            "groupB": "ppmi0",
            "num_bins": 40,
            "alt_hypothesis": "two-sided",
        },
    ),
    "chi_squared": _case(["alzheimerbroadcategory"], ["gender"]),
    "cox_regression_classical": _case(
        ["alzheimerbroadcategory", "lefthippocampus", "leftamygdala"],
        ["timesincebaseline"],
        {"event_var": "alzheimerbroadcategory", "positive_class": "AD"},
        datasets=LONGITUDINAL_DATASETS,
        data_model="longitudinal_dementia:0.1",
        preprocessing=LONGITUDINAL_PREPROCESSING,
    ),
    "cox_regression_stacked": _case(
        ["alzheimerbroadcategory", "lefthippocampus", "leftamygdala"],
        ["timesincebaseline"],
        {
            "event_var": "alzheimerbroadcategory",
            "positive_class": "AD",
            "time_grid_strategy": "distinct_event_times",
            "n_time_bins": 10,
        },
        datasets=LONGITUDINAL_DATASETS,
        data_model="longitudinal_dementia:0.1",
        preprocessing=LONGITUDINAL_PREPROCESSING,
    ),
    "describe": _case(
        [], ["lefthippocampus", "leftamygdala", "alzheimerbroadcategory"], drop_na=False
    ),
    "fisher_exact": _case(
        ["alzheimerbroadcategory"], ["gender"], datasets=["edsd0", "ppmi0"]
    ),
    "glmm_binary": _case(
        ["lefthippocampus", "leftamygdala", "dataset"],
        ["gender"],
        {"positive_class": "F", "grouping_var": ["dataset"]},
    ),
    "glmm_ordinal": _case(
        ["lefthippocampus", "leftamygdala", "dataset"],
        ["agegroup"],
        {
            "grouping_var": ["dataset"],
            "category_order": ["-50y", "50-59y", "60-69y", "70-79y", "+80y"],
        },
    ),
    "histogram": _case(["alzheimerbroadcategory"], ["lefthippocampus"], {"bins": 20}),
    "kmeans": _case([], ["lefthippocampus", "leftamygdala"], {"k": 3}),
    "linear_regression": _case(["leftamygdala", "gender"], ["lefthippocampus"]),
    "linear_regression_cv": _case(
        ["leftamygdala", "gender"], ["lefthippocampus"], {"n_splits": 3}
    ),
    "linear_svm": _case(
        ["lefthippocampus", "leftamygdala"], ["gender"], {"gamma": 0.2, "C": 0.9}
    ),
    "lmm": _case(
        ["lefthippocampus", "leftamygdala", "dataset"],
        ["righthippocampus"],
        {"grouping_var": ["dataset"]},
    ),
    "logistic_regression": _case(
        ["lefthippocampus", "leftamygdala", "alzheimerbroadcategory"],
        ["gender"],
        {"positive_class": "F"},
    ),
    "logistic_regression_cv": _case(
        ["lefthippocampus", "leftamygdala", "alzheimerbroadcategory"],
        ["gender"],
        {"positive_class": "F", "n_splits": 3},
    ),
    "naive_bayes_categorical": _case(
        ["gender", "agegroup"], ["alzheimerbroadcategory"]
    ),
    "naive_bayes_categorical_cv": _case(
        ["gender", "agegroup"], ["alzheimerbroadcategory"], {"n_splits": 3}
    ),
    "naive_bayes_gaussian": _case(
        ["lefthippocampus", "leftamygdala"], ["alzheimerbroadcategory"]
    ),
    "naive_bayes_gaussian_cv": _case(
        ["lefthippocampus", "leftamygdala"], ["alzheimerbroadcategory"], {"n_splits": 3}
    ),
    "pca": _case([], ["lefthippocampus", "righthippocampus", "leftamygdala"]),
    "pearson_correlation": _case(
        [], ["lefthippocampus", "righthippocampus", "leftamygdala"], {"alpha": 0.95}
    ),
    "quartiles": _case([], ["lefthippocampus"], {"num_bins": 20}),
    "standardized_mean_difference": _case(["dataset"], ["lefthippocampus"]),
    "ttest_independent": _case(
        ["dataset"],
        ["lefthippocampus"],
        {
            "groupA": "edsd0",
            "groupB": "ppmi0",
            "alt_hypothesis": "two-sided",
            "alpha": 0.05,
        },
    ),
    "ttest_onesample": _case(
        [],
        ["lefthippocampus"],
        {"mu": 3.0, "alt_hypothesis": "two-sided", "alpha": 0.05},
    ),
    "ttest_paired": _case(
        ["lefthippocampus"],
        ["righthippocampus"],
        {"alt_hypothesis": "two-sided", "alpha": 0.05},
    ),
}

# Aggregation-server round-trips with lazy aggregation; eager counts in comments.
# The remaining round-trips are data dependencies (iterative fits, bisection,
# results checked before the next aggregation) or loop iterations, which are not
# interleaved.
MAX_LAZY_ROUNDS = {
    "anova_oneway": 2,  # eager: 10
    "anova_twoway": 11,  # eager: 37
    "binned_mann_whitney_u_test": 4,  # eager: 8
    "chi_squared": 1,  # eager: 2
    "cox_regression_classical": 8,  # eager: 8
    "cox_regression_stacked": 13,  # eager: 29
    "describe": 38,  # eager: 83
    "fisher_exact": 2,  # eager: 4
    "glmm_binary": 53,  # eager: 54
    "glmm_ordinal": 62,  # eager: 63
    "histogram": 3,  # eager: 6
    "kmeans": 15,  # eager: 26
    "linear_regression": 3,  # eager: 8
    "linear_regression_cv": 13,  # eager: 40
    "linear_svm": 1,  # eager: 1
    "lmm": 714,  # eager: 717
    "logistic_regression": 7,  # eager: 18
    "logistic_regression_cv": 25,  # eager: 79
    "naive_bayes_categorical": 4,  # eager: 4
    "naive_bayes_categorical_cv": 15,  # eager: 15
    "naive_bayes_gaussian": 1,  # eager: 3
    "naive_bayes_gaussian_cv": 6,  # eager: 12
    "pca": 2,  # eager: 4
    "pearson_correlation": 1,  # eager: 6
    "quartiles": 18,  # eager: 36
    "standardized_mean_difference": 31,  # eager: 49
    "ttest_independent": 1,  # eager: 2
    "ttest_onesample": 1,  # eager: 5
    "ttest_paired": 1,  # eager: 3
}


class _InlineRpcError(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self._code = code

    def code(self):
        return self._code


class _InlineGrpcContext:
    def abort(self, code, details):
        raise _InlineRpcError(code, details)


def _link_used_test_data(target: Path) -> Path:
    """Links only the metadata and CSVs the cases use; loading every CSV is slow."""
    used_datasets = {
        "dementia_v_0_1": DATASETS,
        "longitudinal_dementia_v_0_1": LONGITUDINAL_DATASETS,
    }
    for folder, datasets in used_datasets.items():
        (target / folder).mkdir(parents=True)
        for name in ["CDEsMetadata.json", *(f"{code}.csv" for code in datasets)]:
            (target / folder / name).symlink_to(TEST_DATA_PATH / folder / name)
    return target


@pytest.fixture(scope="module")
def worker_duckdb(tmp_path_factory):
    duckdb_path = tmp_path_factory.mktemp("lazy_agg_duckdb") / "data_models.duckdb"
    data_path = _link_used_test_data(tmp_path_factory.mktemp("lazy_agg_data"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(worker_config, "data_path", str(data_path))
        mp.setattr(worker_config.duckdb, "path", str(duckdb_path))
        mp.setattr(duckdb_adapter, "_duckdb_path", duckdb_path)
        load_all_csvs_from_data_folder("lazy-aggregation-tests")
        yield


@pytest.fixture
def aggregation_server(monkeypatch):
    return _install_inline_aggregation_server(monkeypatch)


def _install_inline_aggregation_server(monkeypatch) -> set:
    """Routes aggregation clients to an in-process server; returns seen steps."""
    servicer = AggregationServer()
    steps = set()

    def aggregate(request):
        steps.add((request.request_id, request.step))
        return servicer.Aggregate(request, _InlineGrpcContext())

    # Reduce worker payloads in a canonical order instead of arrival order, so
    # floating-point results do not depend on thread scheduling.
    reduce = servicer._reduce
    servicer._reduce = lambda aggregation_type, vectors: reduce(
        aggregation_type,
        sorted(vectors, key=lambda vector: repr(np.asarray(vector).tolist())),
    )

    stub = SimpleNamespace(
        Configure=lambda request: servicer.Configure(request, _InlineGrpcContext()),
        Unregister=lambda request: servicer.Unregister(request, _InlineGrpcContext()),
        Cleanup=lambda request: servicer.Cleanup(request, _InlineGrpcContext()),
        Aggregate=aggregate,
    )
    monkeypatch.setattr(aggregation_config, "max_wait_for_aggregation_inputs", 30)
    monkeypatch.setattr(
        "exaflow.aggregation_clients.base_aggregation_client.grpc.insecure_channel",
        lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        "exaflow.aggregation_clients.base_aggregation_client.pb2_grpc.AggregationServerStub",
        lambda _channel: stub,
    )
    return steps


class _InProcessEngine:
    """Mirrors Exareme3AlgorithmFlowEngineInterface with in-process workers."""

    def __init__(self, inputdata: Inputdata, metadata: dict, preprocessing: list):
        datasets = sorted(inputdata.datasets)
        self._worker_inputdata = [
            inputdata.model_copy(update={"datasets": datasets[idx::N_WORKERS]})
            for idx in range(N_WORKERS)
            if datasets[idx::N_WORKERS]
        ]
        self._metadata = metadata
        self._preprocessing = preprocessing

    def run_udf(self, func, check_min_rows, add_dataset_variable, kw_args):
        request_id = f"lazy-agg-{uuid.uuid4().hex}"
        controller = ControllerAggregationClient(request_id)
        controller.configure(num_workers=len(self._worker_inputdata))
        wire_kw_args = _struct_to_dict(_dict_to_struct(kw_args))

        def run_worker(inputdata):
            system_args = RunUdfSystemArgs(
                inputdata=inputdata,
                metadata=add_ordered_enums(self._metadata),
                check_min_rows=check_min_rows,
                add_dataset_variable=add_dataset_variable,
                preprocessing=self._preprocessing,
            )
            wire_system_args = RunUdfSystemArgs.model_validate(
                _struct_to_dict(_dict_to_struct(system_args.model_dump()))
            )
            try:
                result = udf_service.run_udf(
                    request_id,
                    get_udf_registry_key(func),
                    copy.deepcopy(wire_kw_args),
                    wire_system_args,
                )
            except InsufficientDataError:
                return None
            return decode_udf_result(encode_udf_result(result))

        try:
            with ThreadPoolExecutor(max_workers=len(self._worker_inputdata)) as pool:
                results = list(pool.map(run_worker, self._worker_inputdata))
        finally:
            controller.cleanup()
            controller.close()
        results = [result for result in results if result is not None]
        if not results:
            raise InsufficientDataError("No worker had sufficient data.")
        return results


def _request(algorithm_name: str, case: dict) -> AnalysisRequestDTO:
    x, y = case["x"], case["y"]
    variables = list(dict.fromkeys([*x, *y]))
    preprocessing = list(case.get("preprocessing") or [])
    if case.get("drop_na", True):
        preprocessing.insert(
            0,
            {
                "name": "missing_values_handler",
                "parameters": {"strategies": {var: "drop" for var in variables}},
            },
        )
    return AnalysisRequestDTO.model_validate(
        {
            "inputdata": {
                "data_model": case.get("data_model", "dementia:0.1"),
                "datasets": case["datasets"],
                "variables": variables,
            },
            "preprocessing": preprocessing,
            "algorithm": {
                "name": algorithm_name,
                "x": x,
                "y": y,
                "parameters": case["parameters"],
            },
        }
    )


def _run_algorithm(algorithm_name: str, request: AnalysisRequestDTO):
    inputdata = Inputdata(**request.inputdata.model_dump())
    cdes = get_data_model_cdes(inputdata.data_model)
    metadata = {
        code: cde.model_dump()
        for code, cde in cdes.values.items()
        if code in inputdata.variables
    }
    # Controller-side preprocessing, as in Exareme3Strategy._run_preprocessing_steps
    transformed_inputdata, transformed_metadata = inputdata, metadata
    for step in request.preprocessing or []:
        preprocessing_step = exareme3_preprocessing_step_classes[step.name](
            params=step.parameters
        )
        preprocessing_step.validate_params(
            inputdata=transformed_inputdata, metadata=transformed_metadata
        )
        transformed_inputdata = transformed_inputdata.model_copy(
            update={
                "variables": preprocessing_step.transform_variables(
                    variables=list(transformed_inputdata.variables)
                )
            }
        )
        transformed_metadata = preprocessing_step.transform_metadata(
            metadata=transformed_metadata
        )
    engine = _InProcessEngine(
        inputdata=inputdata,
        metadata=metadata,
        preprocessing=[step.model_dump() for step in request.preprocessing or []],
    )
    algorithm = exareme3_algorithm_classes[algorithm_name](
        engine=engine,
        logger=Mock(),
        inputdata=transformed_inputdata,
        x=request.algorithm.x,
        y=request.algorithm.y,
        metadata=transformed_metadata,
        parameters=request.algorithm.parameters,
    )
    return json.loads(algorithm.run().model_dump_json())


def _rounds_and_result(algorithm_name, request, steps, monkeypatch, *, lazy: bool):
    with monkeypatch.context() as mp:
        if not lazy:
            mp.setattr(
                udf_service,
                "_wrap_udf_with_lazy_aggregation_if_enabled",
                lambda _key, udf: udf,
            )
        steps.clear()
        result = _run_algorithm(algorithm_name, request)
        return len(steps), result


def test_every_aggregation_algorithm_has_a_lazy_case():
    aggregation_algorithms = {
        name
        for name, cls in exareme3_algorithm_classes.items()
        if ComponentType.AGGREGATION_SERVER in cls.get_specification().components
    }
    assert aggregation_algorithms == set(CASES)
    assert set(MAX_LAZY_ROUNDS) == set(CASES)


@pytest.mark.slow
@pytest.mark.parametrize("algorithm_name", sorted(CASES))
def test_lazy_aggregation_batches_algorithm(
    algorithm_name, worker_duckdb, aggregation_server, monkeypatch
):
    request = _request(algorithm_name, CASES[algorithm_name])

    eager_rounds, eager_result = _rounds_and_result(
        algorithm_name, request, aggregation_server, monkeypatch, lazy=False
    )
    lazy_rounds, lazy_result = _rounds_and_result(
        algorithm_name, request, aggregation_server, monkeypatch, lazy=True
    )

    # Same operations and payloads, canonical server reduction: identical results.
    assert json.dumps(lazy_result) == json.dumps(eager_result)
    assert lazy_rounds <= MAX_LAZY_ROUNDS[algorithm_name] <= eager_rounds
