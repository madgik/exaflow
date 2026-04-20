"""Flower ClientApp runtime for flowertune_llm_medical Phase 1."""

from __future__ import annotations

import json
import os
import time
from math import log2
from typing import Dict

import flwr as fl
import numpy as np
from flwr.common import NDArrays

from exaflow.algorithms.flower.flowertune_llm_medical.controller_io import get_inputdata
from exaflow.algorithms.flower.flowertune_llm_medical.controller_io import (
    get_parameters,
)
from exaflow.algorithms.flower.flowertune_llm_medical.controller_io import get_run_env
from exaflow.algorithms.flower.flowertune_llm_medical.dataset import load_partition
from exaflow.algorithms.flower.flowertune_llm_medical.dataset import load_text_partition
from exaflow.algorithms.flower.flowertune_llm_medical.models import create_backend_model
from exaflow.algorithms.flower.flowertune_llm_medical.run_config import parse_run_config


def _parse_env_json_list(name: str):
    raw = os.getenv(name)
    if not raw:
        return []
    return json.loads(raw)


class FederatedClient(fl.client.NumPyClient):
    def __init__(
        self, model, train_data, eval_data, num_train, num_val, requested_metrics
    ):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.num_train = num_train
        self.num_val = num_val
        self.requested_metrics = requested_metrics

    def get_parameters(self, config):
        _ = config
        return self.model.get_parameters()

    def fit(self, parameters: NDArrays, config: Dict):
        self.model.set_parameters(parameters)
        local_steps = int(os.getenv("LOCAL_STEPS", "1"))
        if isinstance(self.train_data, tuple):
            train_loss = self.model.fit_round(
                self.train_data[0], self.train_data[1], local_steps
            )
        else:
            train_loss = self.model.fit_round(self.train_data, local_steps)
        updated = self.model.get_parameters()
        return updated, self.num_train, {"train_loss": float(train_loss)}

    def evaluate(self, parameters: NDArrays, config: Dict):
        _ = config
        self.model.set_parameters(parameters)
        if isinstance(self.eval_data, tuple):
            metrics = self.model.evaluate_round(self.eval_data[0], self.eval_data[1])
        else:
            metrics = self.model.evaluate_round(self.eval_data)
        reported = {k: v for k, v in metrics.items() if k in self.requested_metrics}
        loss = float(reported.get("loss", metrics.get("loss", 0.0)))
        return loss, self.num_val, reported


def _connect_with_retries(client):
    timeout = max(2, int(os.getenv("TIMEOUT", "30")))
    max_attempts = max(2, int(log2(timeout)) + 1)
    attempts = 0
    while True:
        try:
            fl.client.start_client(
                server_address=os.environ["SERVER_ADDRESS"],
                client=client.to_client(),
            )
            return
        except Exception:
            attempts += 1
            if attempts >= max_attempts:
                raise
            time.sleep(min(2**attempts, 10))


def start_client_app() -> None:
    parameters = get_parameters()
    config = parse_run_config(parameters)
    os.environ.update(get_run_env())

    inputdata = get_inputdata()
    if config.model.backend.value == "hf_peft":
        train_texts, eval_texts = load_text_partition(
            inputdata,
            seed=config.seed,
            val_split_ratio=config.dataset.val_split_ratio,
        )
        n_features = 16
        train_data = train_texts
        eval_data = eval_texts
        num_train = len(train_texts)
        num_val = len(eval_texts)
    else:
        x_train, y_train, x_val, y_val = load_partition(
            inputdata,
            seed=config.seed,
            val_split_ratio=config.dataset.val_split_ratio,
        )
        n_features = int(x_train.shape[1])
        train_data = (x_train, y_train)
        eval_data = (x_val, y_val)
        num_train = len(y_train)
        num_val = len(y_val)

    model = create_backend_model(
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
    requested_metrics = [m.value for m in config.evaluation.metrics]
    # Parse env field to guarantee serialization consistency path is exercised.
    _ = _parse_env_json_list("EVAL_METRICS")

    client = FederatedClient(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        num_train=num_train,
        num_val=num_val,
        requested_metrics=requested_metrics,
    )
    _connect_with_retries(client)
