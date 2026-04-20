"""Qdrant client wrapper for epilepsy RAG."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any
from typing import Iterable

from exaflow.rag.epilepsy.vectorstore.qdrant_schema import COLLECTION_NAME
from exaflow.rag.epilepsy.vectorstore.qdrant_schema import PAYLOAD_INDEX_FIELDS
from exaflow.rag.epilepsy.vectorstore.qdrant_schema import VECTOR_DISTANCE
from exaflow.rag.epilepsy.vectorstore.qdrant_schema import VECTOR_NAME


class VectorStoreBackendError(RuntimeError):
    """Raised when Qdrant dependencies or runtime fail."""


@dataclass(frozen=True)
class QdrantConfig:
    url: str = "http://localhost:6333"
    path: str | None = None
    api_key: str | None = None
    collection_name: str = COLLECTION_NAME
    vector_name: str = VECTOR_NAME
    vector_distance: str = VECTOR_DISTANCE
    embedding_size: int = 768
    timeout: int = 30


def _require_qdrant() -> None:
    if importlib.util.find_spec("qdrant_client") is None:
        raise VectorStoreBackendError(
            "Missing dependency 'qdrant_client' for vector store operations."
        )


class QdrantEpilepsyStore:
    """Thin wrapper around qdrant-client for collection and search operations."""

    def __init__(self, config: QdrantConfig | None = None) -> None:
        self.config = config or QdrantConfig()
        _require_qdrant()

        from qdrant_client import QdrantClient

        self._models = importlib.import_module("qdrant_client.http.models")
        client_kwargs = {"timeout": self.config.timeout}
        if self.config.path:
            client_kwargs["path"] = self.config.path
        else:
            client_kwargs["url"] = self.config.url
            client_kwargs["api_key"] = self.config.api_key
        self._client = QdrantClient(**client_kwargs)

    def ensure_collection(self) -> None:
        models = self._models
        if self._client.collection_exists(self.config.collection_name):
            return
        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={
                self.config.vector_name: models.VectorParams(
                    size=self.config.embedding_size,
                    distance=getattr(models.Distance, self.config.vector_distance.upper()),
                )
            },
        )
        schema_map = {
            "keyword": models.PayloadSchemaType.KEYWORD,
            "integer": models.PayloadSchemaType.INTEGER,
        }
        for field_name, field_type in PAYLOAD_INDEX_FIELDS:
            self._client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name=field_name,
                field_schema=schema_map[field_type],
            )

    def upsert_points(self, points: Iterable[Any]) -> None:
        models = self._models
        normalized_points = [
            point
            if hasattr(point, "id")
            else models.PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point["payload"],
            )
            for point in points
        ]
        self._client.upsert(
            collection_name=self.config.collection_name,
            points=normalized_points,
        )

    def count(self) -> int:
        result = self._client.count(collection_name=self.config.collection_name)
        return int(result.count)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        payload_filter=None,
    ):
        return self._client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            using=self.config.vector_name,
            limit=top_k,
            query_filter=payload_filter,
            with_payload=True,
        )

    def build_filter(self, equals_filters: dict[str, Any]):
        models = self._models
        must = [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
            for key, value in equals_filters.items()
            if value is not None
        ]
        return models.Filter(must=must) if must else None
