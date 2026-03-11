"""Flower ServerApp runtime for flowertune_llm_medical Phase 1."""

from __future__ import annotations

import os

import flwr as fl

from exaflow.algorithms.flower.flowertune_llm_medical.controller_io import get_inputdata
from exaflow.algorithms.flower.flowertune_llm_medical.controller_io import (
    get_parameters,
)
from exaflow.algorithms.flower.flowertune_llm_medical.controller_io import get_run_env
from exaflow.algorithms.flower.flowertune_llm_medical.controller_io import post_result
from exaflow.algorithms.flower.flowertune_llm_medical.dataset import DatasetLoadError
from exaflow.algorithms.flower.flowertune_llm_medical.metrics_out import (
    validate_final_summary_payload,
)
from exaflow.algorithms.flower.flowertune_llm_medical.models import ModelLoadError
from exaflow.algorithms.flower.flowertune_llm_medical.models import (
    initial_parameters_for_backend,
)
from exaflow.algorithms.flower.flowertune_llm_medical.models import preflight_backend
from exaflow.algorithms.flower.flowertune_llm_medical.run_config import parse_run_config
from exaflow.algorithms.flower.flowertune_llm_medical.strategy import build_strategy
from exaflow.algorithms.flower.flowertune_llm_medical.strategy import (
    save_final_checkpoint,
)
from exaflow.algorithms.flower.flowertune_llm_medical.strategy import utc_now


def _job_id() -> str:
    return os.getenv("REQUEST_ID", "unknown")


def _failure_summary(config, code: str, message: str, details: dict):
    payload = {
        "schema_version": "1.1",
        "payload_type": "final_summary",
        "job_id": _job_id(),
        "request_id": _job_id(),
        "algorithm_name": "flowertune_llm_medical",
        "runtime": "simulation",
        "status": "FAILED",
        "timestamp": utc_now(),
        "rounds_total": config.federation.num_rounds,
        "rounds_completed": 0,
        "requested_metrics": [m.value for m in config.evaluation.metrics],
        "reported_metrics": [],
        "aggregate_metrics": {},
        "error": {"code": code, "message": message, "details": details},
    }
    validate_final_summary_payload(payload)
    return payload


def start_server_app() -> None:
    parameters = get_parameters()
    config = parse_run_config(parameters)
    os.environ.update(get_run_env())

    try:
        preflight_backend(config.model.backend.value)
        requested_metrics = [metric.value for metric in config.evaluation.metrics]
        artifact_dir = os.getenv("ARTIFACT_DIR", config.artifacts.artifact_dir)

        strategy, state = build_strategy(
            num_rounds=config.federation.num_rounds,
            fraction_fit=config.federation.fraction_fit,
            fraction_evaluate=config.federation.fraction_evaluate,
            min_fit_clients=config.federation.min_fit_clients,
            min_evaluate_clients=config.federation.min_evaluate_clients,
            min_available_clients=config.federation.min_available_clients,
            checkpoint_every_n_rounds=config.artifacts.checkpoint_every_n_rounds,
            artifact_dir=artifact_dir,
            requested_metrics=requested_metrics,
        )

        inputdata = get_inputdata()
        n_features = len(inputdata.get("x") or []) or 16
        init_arrays = initial_parameters_for_backend(
            backend=config.model.backend.value,
            n_features=n_features,
            learning_rate=config.optimizer.learning_rate,
            model_name=config.model.model_name,
            local_steps=config.local_training.local_steps or 1,
            max_seq_length=config.model.max_seq_length,
            lora_r=config.peft.r or 4,
            lora_alpha=config.peft.alpha or 8,
            lora_dropout=config.peft.dropout or 0.0,
            target_modules=config.peft.target_modules or ["q_proj", "v_proj"],
        )
        strategy.initial_parameters = fl.common.ndarrays_to_parameters(init_arrays)

        fl.server.start_server(
            server_address=os.environ["SERVER_ADDRESS"],
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=config.federation.num_rounds),
        )

        final_checkpoint_ref = ""
        if state.latest_parameters is not None:
            final_checkpoint_ref = save_final_checkpoint(
                artifact_dir, state.latest_parameters
            )
        elif state.checkpoint_refs:
            final_checkpoint_ref = state.checkpoint_refs[-1]
        elif strategy.initial_parameters is not None:
            final_checkpoint_ref = save_final_checkpoint(
                artifact_dir, strategy.initial_parameters
            )

        reported_metrics = sorted(state.last_aggregate_metrics.keys())

        result = {
            "schema_version": "1.1",
            "payload_type": "final_summary",
            "job_id": _job_id(),
            "request_id": _job_id(),
            "algorithm_name": "flowertune_llm_medical",
            "runtime": "simulation",
            "status": "COMPLETED",
            "timestamp": utc_now(),
            "rounds_total": state.rounds_total,
            "rounds_completed": state.rounds_completed,
            "requested_metrics": requested_metrics,
            "reported_metrics": reported_metrics,
            "aggregate_metrics": state.last_aggregate_metrics,
            "clients": {
                "expected": config.federation.num_clients,
                "avg_participated_per_round": state.avg_participated,
            },
            "artifacts": {
                "final_checkpoint_ref": final_checkpoint_ref,
                "intermediate_checkpoints": state.checkpoint_refs,
            },
        }
        if state.best_round is not None:
            result["best_round"] = state.best_round

        validate_final_summary_payload(result)
        post_result(result)

    except DatasetLoadError as exc:
        post_result(
            _failure_summary(
                config,
                code="DATASET_LOAD_ERROR",
                message="Server runtime failed while preparing dataset metadata.",
                details={"exception": str(exc)},
            )
        )
        raise
    except ModelLoadError as exc:
        post_result(
            _failure_summary(
                config,
                code="MODEL_LOAD_ERROR",
                message="Model backend preflight failed.",
                details={"exception": str(exc), "backend": config.model.backend.value},
            )
        )
        raise
    except Exception as exc:  # noqa: BLE001
        post_result(
            _failure_summary(
                config,
                code="RUNTIME_ERROR",
                message="Server runtime failed during Flower execution.",
                details={"exception": str(exc)},
            )
        )
        raise
