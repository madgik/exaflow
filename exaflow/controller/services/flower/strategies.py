import asyncio
import os
from typing import List
from typing import Tuple

import numpy as np

from exaflow import flower_algorithm_folder_paths
from exaflow.algorithms.flower.flowertune_llm_medical.dataset import load_partition
from exaflow.algorithms.flower.flowertune_llm_medical.dataset import load_text_partition
from exaflow.algorithms.flower.flowertune_llm_medical.metrics_out import (
    validate_final_summary_payload,
)
from exaflow.algorithms.flower.flowertune_llm_medical.metrics_out import (
    validate_round_event_payload,
)
from exaflow.algorithms.flower.flowertune_llm_medical.models import ModelLoadError
from exaflow.algorithms.flower.flowertune_llm_medical.models import (
    create_backend_model,
)
from exaflow.algorithms.flower.flowertune_llm_medical.models import preflight_backend
from exaflow.algorithms.flower.flowertune_llm_medical.run_config import (
    build_env_mapping,
)
from exaflow.algorithms.flower.flowertune_llm_medical.run_config import (
    parse_run_config,
)
from exaflow.algorithms.flower.flowertune_llm_medical.run_config import (
    serialize_run_config,
)
from exaflow.algorithms.flower.flowertune_llm_medical.strategy import utc_now
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
            await self._controller.flower_execution_info.reset()
            data_model = self._algorithm_request_dto.inputdata.data_model
            datasets = self._algorithm_request_dto.inputdata.datasets + (
                self._algorithm_request_dto.inputdata.validation_datasets
                if self._algorithm_request_dto.inputdata.validation_datasets
                else []
            )

            self._controller.flower_execution_info.set_inputdata(
                inputdata=self._algorithm_request_dto.inputdata.dict()
            )
            self._controller.flower_execution_info.set_execution_context(
                request_id=self._request_id, algorithm_name=self._algorithm_name
            )
            raw_parameters = self._algorithm_request_dto.parameters or {}
            self._controller.flower_execution_info.set_parameters(raw_parameters)
            self._controller.flower_execution_info.set_run_env({})

            run_config = None
            if self._algorithm_name == "flowertune_llm_medical":
                run_config = parse_run_config(raw_parameters)
                normalized_parameters = serialize_run_config(run_config)
                run_env = build_env_mapping(run_config, self._request_id)
                self._controller.flower_execution_info.set_parameters(
                    normalized_parameters
                )
                self._controller.flower_execution_info.set_run_env(run_env)

                if run_config.runtime.value == "simulation":
                    return await self._execute_flowertune_local_simulation(run_config)

            self._safe_worker_call(
                "garbage collect on global worker",
                self._global_worker_tasks_handler.garbage_collect,
            )
            for handler in self._local_worker_tasks_handlers:
                self._safe_worker_call(
                    f"garbage collect on worker {handler.worker_id}",
                    handler.garbage_collect,
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
                result = (
                    await self._controller.flower_execution_info.get_result_with_timeout()
                )

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

    async def _execute_flowertune_local_simulation(self, run_config):
        try:
            preflight_backend(run_config.model.backend.value)
            inputdata = self._controller.flower_execution_info.get_inputdata()
            num_clients = run_config.federation.num_clients
            num_rounds = run_config.federation.num_rounds
            local_steps = run_config.local_training.local_steps or 1
            requested_metrics = [m.value for m in run_config.evaluation.metrics]

            artifact_dir = run_config.artifacts.artifact_dir.replace(
                "${request_id}", self._request_id
            )
            os.makedirs(artifact_dir, exist_ok=True)

            if run_config.model.backend.value == "tiny":
                client_partitions = []
                for cid in range(num_clients):
                    x_train, y_train, x_val, y_val = load_partition(
                        inputdata,
                        seed=run_config.seed + (cid * 101),
                        val_split_ratio=run_config.dataset.val_split_ratio,
                    )
                    client_partitions.append((x_train, y_train, x_val, y_val))

                n_features = client_partitions[0][0].shape[1]
                template = create_backend_model(
                    backend=run_config.model.backend.value,
                    n_features=n_features,
                    learning_rate=run_config.optimizer.learning_rate,
                    model_name=run_config.model.model_name,
                    local_steps=local_steps,
                    max_seq_length=run_config.model.max_seq_length,
                    lora_r=run_config.peft.r or 4,
                    lora_alpha=run_config.peft.alpha or 8,
                    lora_dropout=run_config.peft.dropout or 0.0,
                    target_modules=run_config.peft.target_modules or ["q_proj", "v_proj"],
                )
            else:
                client_partitions = []
                for cid in range(num_clients):
                    train_texts, val_texts = load_text_partition(
                        inputdata,
                        seed=run_config.seed + (cid * 101),
                        val_split_ratio=run_config.dataset.val_split_ratio,
                    )
                    client_partitions.append((train_texts, val_texts))

                n_features = 1
                template = create_backend_model(
                    backend=run_config.model.backend.value,
                    n_features=n_features,
                    learning_rate=run_config.optimizer.learning_rate,
                    model_name=run_config.model.model_name,
                    local_steps=local_steps,
                    max_seq_length=run_config.model.max_seq_length,
                    lora_r=run_config.peft.r or 4,
                    lora_alpha=run_config.peft.alpha or 8,
                    lora_dropout=run_config.peft.dropout or 0.0,
                    target_modules=run_config.peft.target_modules or ["q_proj", "v_proj"],
                )

            global_params = template.get_parameters()
            best_round = None
            best_loss = None
            final_metrics = {}
            checkpoint_refs = []

            for rnd in range(1, num_rounds + 1):
                local_updates: List[Tuple[List[np.ndarray], int]] = []
                round_eval: List[Tuple[dict, int]] = []

                for cid in range(num_clients):
                    model = create_backend_model(
                        backend=run_config.model.backend.value,
                        n_features=n_features,
                        learning_rate=run_config.optimizer.learning_rate,
                        model_name=run_config.model.model_name,
                        local_steps=local_steps,
                        max_seq_length=run_config.model.max_seq_length,
                        lora_r=run_config.peft.r or 4,
                        lora_alpha=run_config.peft.alpha or 8,
                        lora_dropout=run_config.peft.dropout or 0.0,
                        target_modules=run_config.peft.target_modules or ["q_proj", "v_proj"],
                    )
                    model.set_parameters(global_params)

                    if run_config.model.backend.value == "tiny":
                        x_train, y_train, x_val, y_val = client_partitions[cid]
                        model.fit_round(x_train, y_train, local_steps)
                        metrics = model.evaluate_round(x_val, y_val)
                        local_updates.append((model.get_parameters(), max(1, len(y_train))))
                        round_eval.append((metrics, max(1, len(y_val))))
                    else:
                        train_texts, val_texts = client_partitions[cid]
                        model.fit_round(train_texts, local_steps)
                        metrics = model.evaluate_round(val_texts)
                        local_updates.append(
                            (model.get_parameters(), max(1, len(train_texts)))
                        )
                        round_eval.append((metrics, max(1, len(val_texts))))

                global_params = self._weighted_average_parameters(local_updates)

                total_eval = sum(n for _, n in round_eval)
                agg_loss = (
                    sum(m.get("loss", 0.0) * n for m, n in round_eval)
                    / max(1, total_eval)
                )
                agg_perplexity = (
                    sum(m.get("perplexity", 1.0) * n for m, n in round_eval)
                    / max(1, total_eval)
                )
                final_metrics = {
                    "loss": float(agg_loss),
                    "perplexity": float(agg_perplexity),
                }

                if best_loss is None or agg_loss < best_loss:
                    best_loss = agg_loss
                    best_round = rnd

                checkpoint_ref = None
                if rnd % run_config.artifacts.checkpoint_every_n_rounds == 0:
                    checkpoint_ref = self._save_checkpoint(artifact_dir, rnd, global_params)
                    checkpoint_refs.append(checkpoint_ref)

                round_event = {
                    "schema_version": "1.1",
                    "payload_type": "round_event",
                    "job_id": self._request_id,
                    "request_id": self._request_id,
                    "algorithm_name": "flowertune_llm_medical",
                    "runtime": "simulation",
                    "status": "RUNNING",
                    "round": rnd,
                    "rounds_total": num_rounds,
                    "timestamp": utc_now(),
                    "requested_metrics": requested_metrics,
                    "reported_metrics": sorted(final_metrics.keys()),
                    "metrics": final_metrics,
                    "clients": {
                        "participated": num_clients,
                        "expected": num_clients,
                        "failed": 0,
                    },
                    "artifacts": {"checkpoint_ref": checkpoint_ref},
                }
                validate_round_event_payload(round_event)
                await self._controller.flower_execution_info.add_event(round_event)

            final_checkpoint_ref = self._save_final_checkpoint(artifact_dir, global_params)

            result = {
                "schema_version": "1.1",
                "payload_type": "final_summary",
                "job_id": self._request_id,
                "request_id": self._request_id,
                "algorithm_name": "flowertune_llm_medical",
                "runtime": "simulation",
                "status": "COMPLETED",
                "timestamp": utc_now(),
                "rounds_total": num_rounds,
                "rounds_completed": num_rounds,
                "requested_metrics": requested_metrics,
                "reported_metrics": sorted(final_metrics.keys()),
                "aggregate_metrics": final_metrics,
                "clients": {
                    "expected": num_clients,
                    "avg_participated_per_round": float(num_clients),
                },
                "artifacts": {
                    "final_checkpoint_ref": final_checkpoint_ref,
                    "intermediate_checkpoints": checkpoint_refs,
                },
            }
            if best_round is not None:
                result["best_round"] = best_round

            validate_final_summary_payload(result)
            await self._controller.flower_execution_info.set_result(result)
            return result

        except ModelLoadError as exc:
            return await self._set_failure_summary(
                run_config,
                code="MODEL_LOAD_ERROR",
                message="Model backend preflight failed.",
                details={"exception": str(exc), "backend": run_config.model.backend.value},
            )
        except Exception as exc:  # noqa: BLE001
            return await self._set_failure_summary(
                run_config,
                code="RUNTIME_ERROR",
                message="Local simulation runtime failed.",
                details={"exception": str(exc)},
            )

    async def _set_failure_summary(self, run_config, *, code: str, message: str, details: dict):
        requested_metrics = [m.value for m in run_config.evaluation.metrics]
        result = {
            "schema_version": "1.1",
            "payload_type": "final_summary",
            "job_id": self._request_id,
            "request_id": self._request_id,
            "algorithm_name": "flowertune_llm_medical",
            "runtime": "simulation",
            "status": "FAILED",
            "timestamp": utc_now(),
            "rounds_total": run_config.federation.num_rounds,
            "rounds_completed": 0,
            "requested_metrics": requested_metrics,
            "reported_metrics": [],
            "aggregate_metrics": {},
            "error": {"code": code, "message": message, "details": details},
        }
        validate_final_summary_payload(result)
        await self._controller.flower_execution_info.set_result(result)
        return result

    @staticmethod
    def _weighted_average_parameters(parameter_sets: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        total_examples = sum(num_examples for _, num_examples in parameter_sets)
        if total_examples <= 0:
            raise ValueError("Cannot aggregate parameters with zero examples")
        avg_params = None
        for params, num_examples in parameter_sets:
            weight = num_examples / total_examples
            if avg_params is None:
                avg_params = [weight * p for p in params]
            else:
                for i, p in enumerate(params):
                    avg_params[i] += weight * p
        return avg_params

    @staticmethod
    def _save_checkpoint(artifact_dir: str, rnd: int, parameters: List[np.ndarray]) -> str:
        os.makedirs(artifact_dir, exist_ok=True)
        ckpt_path = os.path.join(artifact_dir, f"round_{rnd:03d}.npz")
        np.savez(ckpt_path, **{f"tensor_{i:04d}": arr for i, arr in enumerate(parameters)})
        return ckpt_path

    @staticmethod
    def _save_final_checkpoint(artifact_dir: str, parameters: List[np.ndarray]) -> str:
        os.makedirs(artifact_dir, exist_ok=True)
        ckpt_path = os.path.join(artifact_dir, "final_model.npz")
        np.savez(ckpt_path, **{f"tensor_{i:04d}": arr for i, arr in enumerate(parameters)})
        return ckpt_path

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
