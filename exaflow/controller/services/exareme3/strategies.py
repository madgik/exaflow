from typing import List

from exaflow import exareme3_algorithm_classes
from exaflow import exareme3_preprocessing_step_classes
from exaflow.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.controller import config as controller_config
from exaflow.controller.federation_info_logs import log_experiment_execution
from exaflow.controller.services.exareme3 import Exareme3Controller
from exaflow.controller.services.exareme3.algorithm_flow_engine_interface import (
    Exareme3AlgorithmFlowEngineInterface,
)
from exaflow.controller.services.exareme3.tasks_handler import Exareme3TasksHandler
from exaflow.controller.services.strategy_interface import AlgorithmExecutionStrategyI
from exaflow.protos.aggregation_server import aggregation_server_pb2 as agg_pb2


class Exareme3Strategy(AlgorithmExecutionStrategyI):
    _controller: Exareme3Controller
    _local_worker_tasks_handlers: List[Exareme3TasksHandler]
    _global_worker_tasks_handler: Exareme3TasksHandler

    @staticmethod
    def _run_preprocessing_steps(
        inputdata: Inputdata,
        metadata: dict,
        preprocessing: dict,
    ) -> Inputdata:
        transformed_inputdata = inputdata
        transformed_metadata = metadata
        for preprocessing_step_name, preprocessing_step_params in preprocessing.items():
            preprocessing_step_cls = exareme3_preprocessing_step_classes[
                preprocessing_step_name
            ]
            step_inputdata = transformed_inputdata
            step_metadata = transformed_metadata
            preprocessing_step = preprocessing_step_cls(
                params=preprocessing_step_params,
            )
            preprocessing_step.validate_params(
                inputdata=step_inputdata,
                metadata=step_metadata,
            )
            transformed_x, transformed_y = (
                preprocessing_step.transform_inputdata_variables(
                    x=list(step_inputdata.x or []),
                    y=list(step_inputdata.y or []),
                )
            )
            transformed_inputdata = step_inputdata.model_copy(
                update={"x": transformed_x, "y": transformed_y}
            )
            transformed_metadata = preprocessing_step.transform_metadata(
                metadata=step_metadata,
            )
        return transformed_inputdata

    async def execute(self) -> str:
        inputdata = Inputdata.model_validate(
            self._algorithm_request_dto.inputdata.model_dump()
        )
        variable_names = (inputdata.x or []) + (inputdata.y or [])
        metadata = self._controller.worker_landscape_aggregator.get_metadata(
            data_model=inputdata.data_model,
            variable_names=variable_names,
        )

        preprocessing = self._algorithm_request_dto.preprocessing or {}
        transformed_inputdata = self._run_preprocessing_steps(
            inputdata=inputdata,
            metadata=metadata,
            preprocessing=preprocessing,
        )

        engine = Exareme3AlgorithmFlowEngineInterface(
            request_id=self._request_id,
            context_id=self._context_id,
            tasks_handlers=self._local_worker_tasks_handlers,
            inputdata=inputdata,
            metadata=metadata,
            preprocessing=preprocessing,
        )
        algorithm_cls = exareme3_algorithm_classes[self._algorithm_name]
        algorithm = algorithm_cls(
            engine=engine,
            inputdata=transformed_inputdata,
            parameters=self._algorithm_request_dto.parameters,
        )
        log_experiment_execution(
            self._logger,
            self._request_id,
            self._context_id,
            self._algorithm_name,
            transformed_inputdata.datasets,
            self._algorithm_request_dto.parameters,
            [h.worker_id for h in self._local_worker_tasks_handlers],
        )
        result = algorithm.run()
        self._logger.info(
            f"Execution completed: {self._algorithm_name} ({self._request_id})"
        )
        return result.model_dump_json()


class Exareme3WithAggregationServerStrategy(Exareme3Strategy):
    async def execute(self) -> str:
        agg_dns = (
            getattr(getattr(controller_config, "aggregation_server", {}), "dns", None)
            or None
        )
        agg_client = ControllerAggregationClient(
            self._request_id, aggregator_dns=agg_dns
        )
        status = agg_client.configure(
            num_workers=len(self._local_worker_tasks_handlers)
        )
        if status != agg_pb2.Status.OK:
            raise RuntimeError(f"AggregationServer refused to configure: {status}")
        self._logger.debug(f"Aggregation configured: {status}")

        try:
            return await super().execute()
        finally:
            cleanup_status = agg_client.cleanup()
            self._logger.debug(f"Aggregation cleanup response: {cleanup_status}")
