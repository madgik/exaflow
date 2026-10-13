import asyncio
import logging
from threading import Event

from exaflow import feta_algorithm_classes
from exaflow.algorithms.feta.columns import ColumnMetadata
from exaflow.controller import config
from exaflow.controller.services import get_worker_landscape_aggregator
from exaflow.controller.services.feta.runtime import FetaRuntime
from exaflow.controller.services.strategy_interface import AlgorithmExecutionStrategyI
from exaflow.feta.clients import FetaAggregationClient
from exaflow.feta.clients import FetaWorkerClient
from exaflow.feta.inputdata import Inputdata
from exaflow.feta.lifecycle import ContextClosedError
from exaflow.feta.settings import FetaSettings
from exaflow.worker_communication import BadUserInput


class FetaStrategy(AlgorithmExecutionStrategyI):
    def __init__(self, analysis_request_dto):
        self.request = analysis_request_dto
        self.cancelled = Event()
        self.runtime = None

    async def execute(self):
        # Keep the Quart event loop available while synchronous gRPC calls run.
        work = asyncio.create_task(asyncio.to_thread(self._execute))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            self.cancelled.set()
            if self.runtime is not None:
                await asyncio.to_thread(self.runtime.close, True)
            try:
                await work
            except Exception as exc:
                logging.getLogger(__name__).info(
                    "FETA execution terminated after cancellation (%s).",
                    type(exc).__name__,
                )
            raise

    def _execute(self):
        if self.cancelled.is_set():
            raise ContextClosedError("FETA execution was cancelled.")
        request = self.request
        settings = FetaSettings.from_config(config)
        if request.preprocessing:
            raise BadUserInput("FETA does not support preprocessing.")
        if request.inputdata.validation_datasets:
            raise BadUserInput("FETA does not yet support validation datasets.")
        if request.flags and any(request.flags.values()):
            raise BadUserInput("FETA does not support optional execution flags.")
        algorithm_class = feta_algorithm_classes[request.algorithm.name]
        specification = algorithm_class.get_specification()
        columns = {
            name: list(getattr(request.algorithm, name))
            for name in ("x", "y")
            if getattr(specification, name) is not None
            and getattr(request.algorithm, name)
        }
        algorithm = algorithm_class(
            parameters=request.algorithm.parameters,
            x_names=columns.get("x", []),
            y_names=columns.get("y", []),
        )
        inputdata = Inputdata(**request.inputdata.model_dump())
        landscape = get_worker_landscape_aggregator()
        ids = landscape.get_worker_ids_with_any_of_datasets(
            request.inputdata.data_model, request.inputdata.datasets
        )
        timeout = int(config.worker_tasks_timeout or 120)
        infos = [
            (worker_id, landscape.get_worker_info(worker_id))
            for worker_id in sorted(ids)
        ]
        aggregation = FetaAggregationClient(config.aggregation_server.dns, timeout)
        workers = []
        try:
            for worker_id, info in infos:
                workers.append(
                    FetaWorkerClient(worker_id, f"{info.ip}:{info.port}", timeout)
                )
        except Exception:
            for worker in workers:
                worker.close()
            aggregation.close()
            raise
        with FetaRuntime(workers, aggregation, settings=settings) as runtime:
            self.runtime = runtime
            if self.cancelled.is_set():
                raise ContextClosedError("FETA execution was cancelled.")
            column_groups = algorithm.column_inputs.intersection(columns)
            metadata = {}
            if column_groups:
                cdes = landscape.get_cdes(request.inputdata.data_model)
                for group in column_groups:
                    for name in columns[group]:
                        cde = cdes[name]
                        enums = cde.enumerations or {}
                        converter = {"int": int, "real": float, "text": str}[
                            cde.sql_type
                        ]
                        pairs = sorted(
                            (converter(code), label) for code, label in enums.items()
                        )
                        metadata[name] = ColumnMetadata(
                            cde.sql_type,
                            cde.is_categorical,
                            tuple(code for code, _ in pairs),
                            tuple(str(label) for _, label in pairs),
                        )
            tensors = runtime.initialize(inputdata, columns, column_groups, metadata)
            result = algorithm.run(**tensors)
            runtime.check_active()
            return result.model_dump_json()
