"""MedCPT embedding wrapper for epilepsy RAG."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np


DEFAULT_QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
DEFAULT_ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"


class EmbeddingBackendError(RuntimeError):
    """Raised when embedding dependencies or runtime fail."""


@dataclass(frozen=True)
class MedCPTConfig:
    query_model_name: str = DEFAULT_QUERY_MODEL_NAME
    article_model_name: str = DEFAULT_ARTICLE_MODEL_NAME
    max_length: int = 512
    batch_size: int = 8
    device: str = "cpu"


@dataclass(frozen=True)
class HashEmbeddingConfig:
    dimension: int = 384


def _require_dependency(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise EmbeddingBackendError(
            f"Missing dependency '{module_name}' for MedCPT embeddings."
        )


class MedCPTEmbedder:
    """Dense embedder using the MedCPT article/query encoder pair."""

    def __init__(self, config: MedCPTConfig | None = None) -> None:
        self.config = config or MedCPTConfig()
        _require_dependency("torch")
        _require_dependency("transformers")

        import torch
        from transformers import AutoModel
        from transformers import AutoTokenizer

        self._torch = torch
        self._query_tokenizer = AutoTokenizer.from_pretrained(
            self.config.query_model_name
        )
        self._query_model = AutoModel.from_pretrained(self.config.query_model_name)
        self._article_tokenizer = AutoTokenizer.from_pretrained(
            self.config.article_model_name
        )
        self._article_model = AutoModel.from_pretrained(self.config.article_model_name)

        if self.config.device != "cpu":
            self._query_model.to(self.config.device)
            self._article_model.to(self.config.device)
        self._query_model.eval()
        self._article_model.eval()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(
            texts=texts,
            tokenizer=self._article_tokenizer,
            model=self._article_model,
        )

    def embed_query(self, query: str) -> list[float]:
        vectors = self._embed(
            texts=[query],
            tokenizer=self._query_tokenizer,
            model=self._query_model,
        )
        return vectors[0]

    def embedding_size(self) -> int:
        probe = self.embed_query("epilepsy")
        return len(probe)

    def _embed(self, texts, tokenizer, model) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.config.batch_size):
                batch = list(texts[start : start + self.config.batch_size])
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                if self.config.device != "cpu":
                    encoded = {
                        key: value.to(self.config.device)
                        for key, value in encoded.items()
                    }
                outputs = model(**encoded)
                pooled = outputs.last_hidden_state[:, 0, :]
                normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(normalized.cpu().numpy().astype(np.float32).tolist())
        return vectors


class HashingEmbedder:
    """Offline deterministic fallback embedder for smoke retrieval."""

    def __init__(self, config: HashEmbeddingConfig | None = None) -> None:
        self.config = config or HashEmbeddingConfig()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_text(query)

    def embedding_size(self) -> int:
        return self.config.dimension

    def _embed_text(self, text: str) -> list[float]:
        vector = np.zeros((self.config.dimension,), dtype=np.float32)
        for token in text.lower().split():
            digest = sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.config.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


def build_embedder(backend: str, *, device: str = "cpu"):
    if backend == "medcpt":
        return MedCPTEmbedder(MedCPTConfig(device=device))
    if backend == "hash":
        return HashingEmbedder()
    if backend == "auto":
        try:
            return MedCPTEmbedder(MedCPTConfig(device=device))
        except Exception:
            return HashingEmbedder()
    raise ValueError(f"Unsupported embedding backend '{backend}'")
