import asyncio
from typing import List

from exaflow import flower_algorithm_folder_paths
from exaflow.controller import config as ctrl_config
from exaflow.controller.federation_info_logs import log_experiment_execution
from exaflow.controller.services.errors import WorkerTaskTimeoutError
from exaflow.controller.services.flower import FlowerController
from exaflow.controller.services.flower.tasks_handler import FlowerTasksHandler
from exaflow.controller.services.strategy_interface import AlgorithmExecutionStrategyI
from exaflow.controller.worker_client.app import WorkerClientConnectionError
from exaflow.controller.worker_client.app import WorkerClientTimeoutException


class FlowerStrategy(AlgorithmExecutionStrategyI):
    _controller: FlowerController
    _local_worker_tasks_handlers: List[FlowerTasksHandler]
    _global_worker_tasks_handler: FlowerTasksHandler

    async def execute(self) -> str:
        async with self._controller.algorithm_execution_lock:
            data_model = self._algorithm_request_dto.inputdata.data_model
            datasets = self._algorithm_request_dto.inputdata.datasets + (
                self._algorithm_request_dto.inputdata.validation_datasets
                if self._algorithm_request_dto.inputdata.validation_datasets
                else []
            )

            self._safe_worker_call(
                "garbage collect on global worker",
                self._global_worker_tasks_handler.garbage_collect,
            )
            for handler in self._local_worker_tasks_handlers:
                self._safe_worker_call(
                    f"garbage collect on worker {handler.worker_id}",
                    handler.garbage_collect,
                )

            self._controller.flower_execution_info.set_inputdata(
                inputdata=self._algorithm_request_dto.inputdata.model_dump()
            )
            server_pid = None
            clients_pids = {}
            server_address = f"{self._controller.worker_landscape_aggregator.get_global_worker().ip}:{ctrl_config.flower.server_port}"
            algorithm_folder_path = flower_algorithm_folder_paths[self._algorithm_name]
            try:
                server_pid = self._global_worker_tasks_handler.start_flower_server(
                    algorithm_folder_path,
                    len(self._local_worker_tasks_handlers),
                    str(server_address),
                    data_model,
                    datasets,
                )
                clients_pids = {
                    handler.start_flower_client(
                        algorithm_folder_path,
                        str(server_address),
                        data_model,
                        datasets,
                        ctrl_config.flower.execution_timeout,
                    ): handler
                    for handler in self._local_worker_tasks_handlers
                }

                log_experiment_execution(
                    self._logger,
                    self._request_id,
                    self._context_id,
                    self._algorithm_name,
                    self._algorithm_request_dto.inputdata.datasets,
                    self._algorithm_request_dto.parameters,
                    [h.worker_id for h in self._local_worker_tasks_handlers],
                )
                result = await self._controller.flower_execution_info.get_result_with_timeout()

                self._logger.info(
                    f"Finished execution -> {self._algorithm_name} with {self._request_id}"
                )

                return result

            except asyncio.TimeoutError:
                raise WorkerTaskTimeoutError()
            finally:
                await self._cleanup(
                    self._algorithm_name,
                    self._global_worker_tasks_handler,
                    server_pid,
                    clients_pids,
                )

    def _safe_worker_call(self, action_desc, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except (WorkerClientTimeoutException, WorkerClientConnectionError) as exc:
            self._logger.warning("Failed to %s: %s", action_desc, exc)
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Unexpected error while %s: %s", action_desc, exc)

    async def _cleanup(
        self, algorithm_name, server_task_handler, server_pid, clients_pids
    ):
        await self._controller.flower_execution_info.reset()
        if server_pid is not None:
            self._safe_worker_call(
                f"stop flower server pid={server_pid}",
                server_task_handler.stop_flower_server,
                server_pid,
                algorithm_name,
            )
        for pid, handler in clients_pids.items():
            self._safe_worker_call(
                f"stop flower client pid={pid} on worker {handler.worker_id}",
                handler.stop_flower_client,
                pid,
                algorithm_name,
            )
