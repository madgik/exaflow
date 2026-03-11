"""Play mode entrypoint for local federated training simulation.

Run from IDE "Play" to observe round-by-round progress and final metrics
without starting the full Exaflow controller/worker stack.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np

# Allow running both as module and direct script file.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from exaflow.algorithms.flower.flowertune_llm_medical.dataset import load_partition
from exaflow.algorithms.flower.flowertune_llm_medical.models import ModelLoadError
from exaflow.algorithms.flower.flowertune_llm_medical.models import (
    create_backend_model,
)
from exaflow.algorithms.flower.flowertune_llm_medical.models import preflight_backend
from exaflow.algorithms.flower.flowertune_llm_medical.run_config import parse_run_config


def _build_run_config_from_args(args) -> dict:
    return {
        "schema_version": "1.1",
        "runtime": "simulation",
        "seed": args.seed,
        "federation": {
            "num_rounds": args.rounds,
            "num_clients": args.clients,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "min_fit_clients": args.clients,
            "min_evaluate_clients": args.clients,
            "min_available_clients": args.clients,
        },
        "model": {
            "backend": args.backend,
            "model_name": args.model_name,
            "task_type": "causal_lm",
            "max_seq_length": 128,
            "quantization": "none",
        },
        "peft": {
            "enabled": True,
            "method": "lora",
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["q_proj"],
        },
        "optimizer": {
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
        "local_training": {
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "local_steps": args.local_steps,
            "fp16": False,
            "bf16": True,
        },
        "dataset": {
            "dataset_name": "synthetic",
            "split": "train",
            "partitioner": "iid",
            "num_partitions": args.clients,
            "partition_id_strategy": "flower_simulation",
            "val_split_ratio": args.val_split_ratio,
        },
        "evaluation": {
            "enabled": True,
            "evaluate_every_n_rounds": 1,
            "metrics": ["loss", "perplexity"],
        },
        "artifacts": {
            "artifact_dir": args.artifact_dir,
            "checkpoint_every_n_rounds": 1,
            "save_final_model": True,
            "save_optimizer_state": False,
        },
    }


def _weighted_average_parameters(
    parameter_sets: List[Tuple[List[np.ndarray], int]],
) -> List[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in parameter_sets)
    if total_examples <= 0:
        raise ValueError("Cannot aggregate parameters with zero examples.")
    avg_params = None
    for params, num_examples in parameter_sets:
        weight = num_examples / total_examples
        if avg_params is None:
            avg_params = [weight * p for p in params]
        else:
            for i, p in enumerate(params):
                avg_params[i] += weight * p
    return avg_params


def _save_play_checkpoint(path: Path, parameters: List[np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weights=parameters[0], bias=parameters[1])
    return path


def run_play_mode(args) -> int:
    raw_cfg = _build_run_config_from_args(args)
    config = parse_run_config(raw_cfg)

    try:
        preflight_backend(config.model.backend.value)
    except ModelLoadError as exc:
        print(f"[ERROR] backend preflight failed: {exc}")
        return 1

    # Build synthetic local partitions for each simulated client.
    client_partitions = []
    for cid in range(config.federation.num_clients):
        x_train, y_train, x_val, y_val = load_partition(
            {"x": [], "y": []},
            seed=config.seed + (cid * 101),
            val_split_ratio=config.dataset.val_split_ratio,
        )
        client_partitions.append((x_train, y_train, x_val, y_val))

    n_features = client_partitions[0][0].shape[1]
    template_model = create_backend_model(
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
    global_params = template_model.get_parameters()

    best_round = None
    best_loss = None
    final_metrics = {"loss": 0.0, "perplexity": 0.0}

    print(
        f"[START] backend={config.model.backend.value} rounds={config.federation.num_rounds} clients={config.federation.num_clients}"
    )

    for rnd in range(1, config.federation.num_rounds + 1):
        local_updates = []
        round_eval = []
        for cid, (x_train, y_train, x_val, y_val) in enumerate(client_partitions):
            client_model = create_backend_model(
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
            client_model.set_parameters(global_params)
            train_loss = client_model.fit_round(
                x_train, y_train, config.local_training.local_steps
            )
            local_updates.append((client_model.get_parameters(), len(y_train)))
            metrics = client_model.evaluate_round(x_val, y_val)
            round_eval.append((metrics, len(y_val)))
            print(
                f"[ROUND {rnd}] client={cid} train_loss={train_loss:.6f} val_loss={metrics['loss']:.6f}"
            )

        global_params = _weighted_average_parameters(local_updates)

        total_val = sum(n for _, n in round_eval)
        agg_loss = sum(m["loss"] * n for m, n in round_eval) / max(1, total_val)
        agg_ppl = sum(m["perplexity"] * n for m, n in round_eval) / max(1, total_val)
        final_metrics = {"loss": float(agg_loss), "perplexity": float(agg_ppl)}

        if best_loss is None or agg_loss < best_loss:
            best_loss = agg_loss
            best_round = rnd

        print(
            f"[ROUND {rnd}] aggregate loss={agg_loss:.6f} perplexity={agg_ppl:.6f}"
        )

    ckpt_path = _save_play_checkpoint(
        Path(config.artifacts.artifact_dir) / "play_final_model.npz", global_params
    )
    print("[DONE] Training completed")
    print(
        f"[RESULT] rounds={config.federation.num_rounds} best_round={best_round} best_loss={best_loss:.6f}"
    )
    print(
        f"[RESULT] final_loss={final_metrics['loss']:.6f} final_perplexity={final_metrics['perplexity']:.6f}"
    )
    print(f"[RESULT] final_checkpoint_ref={ckpt_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Play mode training runner.")
    parser.add_argument("--backend", choices=["tiny", "hf_peft"], default="tiny")
    parser.add_argument("--model-name", default="tiny")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split-ratio", type=float, default=0.2)
    parser.add_argument("--artifact-dir", default="runs/play-mode")
    args = parser.parse_args()
    raise SystemExit(run_play_mode(args))


if __name__ == "__main__":
    main()
