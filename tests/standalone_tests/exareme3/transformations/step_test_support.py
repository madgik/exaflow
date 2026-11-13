"""Focused multi-participant execution of the existing preprocessing contract."""

from concurrent.futures import ThreadPoolExecutor

from exaflow.algorithms.utils.inputdata_utils import Inputdata
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def metadata(name="crp", sql_type="real", categorical=False):
    return {
        name: {
            "code": name,
            "label": name,
            "sql_type": sql_type,
            "is_categorical": categorical,
            "min": 0,
            "max": 100,
        }
    }


def validate(step, current_metadata):
    step.validate_params(
        inputdata=Inputdata(
            data_model="dm:0.1", datasets=["d1"], variables=list(current_metadata)
        ),
        metadata=current_metadata,
    )


def run_workers(step_class, params, frames, current_metadata):
    validate(step_class(params=params), current_metadata)
    coordinator = AggregationCoordinator(len(frames))

    def one(index):
        step = step_class(params=params)
        try:
            result = step.transform_data_and_metadata(
                data=frames[index].copy(deep=True),
                metadata=current_metadata,
                agg_client=SimulatedAggClient(worker_id=index, coordinator=coordinator),
            )
            return step, result
        except Exception as exc:
            coordinator.abort(exc)
            raise

    with ThreadPoolExecutor(max_workers=len(frames)) as executor:
        futures = [executor.submit(one, i) for i in range(len(frames))]
        return [future.result(timeout=30) for future in futures]
