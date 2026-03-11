"""Model backend implementations for flowertune_llm_medical."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Dict
from typing import List

import numpy as np


class ModelLoadError(RuntimeError):
    """Raised when requested model backend cannot be initialized."""


@dataclass
class TinyAdapterModel:
    """A small logistic adapter model to exercise federated runtime wiring."""

    n_features: int
    learning_rate: float

    def __post_init__(self):
        self.weights = np.zeros((self.n_features,), dtype=np.float32)
        self.bias = np.zeros((1,), dtype=np.float32)

    def get_parameters(self):
        return [self.weights.copy(), self.bias.copy()]

    def set_parameters(self, params):
        self.weights = np.asarray(params[0], dtype=np.float32).reshape(self.n_features)
        self.bias = np.asarray(params[1], dtype=np.float32).reshape(1)

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights + self.bias[0]
        logits = np.clip(logits, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    @staticmethod
    def _binary_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-7
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return float(
            -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        )

    def fit_round(self, x: np.ndarray, y: np.ndarray, local_steps: int) -> float:
        n = max(1, x.shape[0])
        for _ in range(local_steps):
            probs = self._predict_proba(x)
            err = probs - y
            grad_w = (x.T @ err) / n
            grad_b = np.mean(err)
            self.weights -= self.learning_rate * grad_w.astype(np.float32)
            self.bias[0] -= self.learning_rate * float(grad_b)
        final_prob = self._predict_proba(x)
        return self._binary_loss(y, final_prob)

    def evaluate_round(self, x: np.ndarray, y: np.ndarray):
        prob = self._predict_proba(x)
        loss = self._binary_loss(y, prob)
        perplexity = float(np.exp(min(20.0, loss)))
        return {"loss": loss, "perplexity": perplexity}


class HFPeftAdapterModel:
    """HF+PEFT backend exchanging only adapter tensors."""

    def __init__(
        self,
        *,
        model_name: str,
        learning_rate: float,
        local_steps: int,
        max_seq_length: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: List[str],
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.local_steps = local_steps
        self.max_seq_length = max_seq_length

        self.torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")

        self.AutoTokenizer = transformers.AutoTokenizer
        self.AutoModelForCausalLM = transformers.AutoModelForCausalLM
        self.LoraConfig = peft.LoraConfig
        self.get_peft_model = peft.get_peft_model
        self.get_peft_model_state_dict = peft.get_peft_model_state_dict
        self.set_peft_model_state_dict = peft.set_peft_model_state_dict

        self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = self.AutoModelForCausalLM.from_pretrained(self.model_name)
        peft_config = self.LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        self.model = self.get_peft_model(base_model, peft_config)
        self.model.train()
        self.optimizer = self.torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        adapter_state = self.get_peft_model_state_dict(self.model)
        self.adapter_keys = sorted(adapter_state.keys())

    def _tokenize(self, texts: List[str]):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    def get_parameters(self):
        state = self.get_peft_model_state_dict(self.model)
        params = []
        for key in self.adapter_keys:
            params.append(state[key].detach().cpu().numpy().astype(np.float32))
        return params

    def set_parameters(self, params):
        state = self.get_peft_model_state_dict(self.model)
        for key, arr in zip(self.adapter_keys, params):
            tensor = self.torch.tensor(arr)
            tensor = tensor.to(dtype=state[key].dtype)
            state[key] = tensor
        self.set_peft_model_state_dict(self.model, state)

    def fit_round(self, train_texts: List[str], local_steps: int) -> float:
        if not train_texts:
            return 0.0
        self.model.train()
        losses = []
        batch_size = min(4, max(1, len(train_texts)))
        for step in range(local_steps):
            start = (step * batch_size) % len(train_texts)
            batch = train_texts[start : start + batch_size]
            if not batch:
                batch = train_texts[:batch_size]
            inputs = self._tokenize(batch)
            outputs = self.model(**inputs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else 0.0

    def evaluate_round(self, eval_texts: List[str]):
        if not eval_texts:
            return {"loss": 0.0, "perplexity": 1.0}
        self.model.eval()
        with self.torch.no_grad():
            batch = eval_texts[: min(8, len(eval_texts))]
            inputs = self._tokenize(batch)
            outputs = self.model(**inputs)
            loss = float(outputs.loss.detach().cpu().item())
        perplexity = float(np.exp(min(20.0, loss)))
        return {"loss": loss, "perplexity": perplexity}


def create_model(n_features: int, learning_rate: float) -> TinyAdapterModel:
    return TinyAdapterModel(n_features=n_features, learning_rate=learning_rate)


def _ensure_hf_peft_dependencies():
    missing = [
        package
        for package in ("torch", "transformers", "peft")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise ModelLoadError(
            "hf_peft backend requires missing dependencies: "
            + ", ".join(sorted(missing))
        )


def preflight_backend(backend: str) -> None:
    """Validate backend dependencies before runtime starts."""
    if backend == "hf_peft":
        _ensure_hf_peft_dependencies()
    elif backend != "tiny":
        raise ModelLoadError(f"Unsupported model backend: {backend}")


def create_backend_model(
    *,
    backend: str,
    n_features: int,
    learning_rate: float,
    model_name: str,
    local_steps: int,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
):
    if backend == "tiny":
        return create_model(n_features=n_features, learning_rate=learning_rate)
    if backend == "hf_peft":
        _ensure_hf_peft_dependencies()
        return HFPeftAdapterModel(
            model_name=model_name,
            learning_rate=learning_rate,
            local_steps=local_steps,
            max_seq_length=max_seq_length,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
    raise ModelLoadError(f"Unsupported model backend: {backend}")


def initial_parameters_for_backend(
    *,
    backend: str,
    n_features: int,
    learning_rate: float,
    model_name: str,
    local_steps: int,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
):
    model = create_backend_model(
        backend=backend,
        n_features=n_features,
        learning_rate=learning_rate,
        model_name=model_name,
        local_steps=local_steps,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    return model.get_parameters()
