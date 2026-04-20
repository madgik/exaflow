"""Flower strategy helpers for flowertune_llm_medical Phase 1 runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from flwr.common import parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class StrategyState:
    rounds_total: int
    requested_metrics: List[str]
    last_aggregate_metrics: Dict[str, float]
    rounds_completed: int
    avg_participated: float
    best_round: Optional[int]
    best_loss: Optional[float]
    checkpoint_refs: List[str]
    latest_parameters: Optional[object]


class ContractFedAvg(FedAvg):
    """FedAvg strategy that records summary stats and checkpoints."""

    def __init__(
        self,
        state: StrategyState,
        checkpoint_every_n_rounds: int,
        artifact_dir: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state = state
        self.checkpoint_every_n_rounds = checkpoint_every_n_rounds
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def _save_checkpoint(self, rnd: int, parameters) -> None:
        arrays = parameters_to_ndarrays(parameters)
        ckpt_path = os.path.join(self.artifact_dir, f"round_{rnd:03d}.npz")
        np.savez(ckpt_path, **{f"tensor_{i:04d}": arr for i, arr in enumerate(arrays)})
        self.state.checkpoint_refs.append(ckpt_path)

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is None:
            return None
        parameters, fit_metrics = aggregated
        self.state.latest_parameters = parameters
        self.state.rounds_completed = rnd
        if results:
            avg_participated_prev = self.state.avg_participated * max(0, rnd - 1)
            self.state.avg_participated = (avg_participated_prev + len(results)) / rnd
        if rnd % self.checkpoint_every_n_rounds == 0:
            self._save_checkpoint(rnd, parameters)
        return parameters, fit_metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[tuple[ClientProxy, object]],
        failures,
    ):
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        if aggregated is None:
            return None
        loss, metrics = aggregated
        merged = {"loss": float(loss)}
        merged.update({k: float(v) for k, v in metrics.items()})
        self.state.last_aggregate_metrics = merged
        cur_loss = merged.get("loss")
        if cur_loss is not None and (
            self.state.best_loss is None or cur_loss < self.state.best_loss
        ):
            self.state.best_loss = cur_loss
            self.state.best_round = rnd
        return aggregated


def build_strategy(
    *,
    num_rounds: int,
    fraction_fit: float,
    fraction_evaluate: float,
    min_fit_clients: int,
    min_evaluate_clients: int,
    min_available_clients: int,
    checkpoint_every_n_rounds: int,
    artifact_dir: str,
    requested_metrics: List[str],
):
    state = StrategyState(
        rounds_total=num_rounds,
        requested_metrics=requested_metrics,
        last_aggregate_metrics={},
        rounds_completed=0,
        avg_participated=0.0,
        best_round=None,
        best_loss=None,
        checkpoint_refs=[],
        latest_parameters=None,
    )

    strategy = ContractFedAvg(
        state=state,
        checkpoint_every_n_rounds=checkpoint_every_n_rounds,
        artifact_dir=artifact_dir,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    return strategy, state


def save_final_checkpoint(artifact_dir: str, parameters) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    arrays = parameters_to_ndarrays(parameters)
    ckpt_path = os.path.join(artifact_dir, "final_model.npz")
    np.savez(ckpt_path, **{f"tensor_{i:04d}": arr for i, arr in enumerate(arrays)})
    return ckpt_path
