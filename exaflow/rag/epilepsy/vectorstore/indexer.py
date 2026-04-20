"""Indexing helpers for epilepsy RAG chunks."""

from __future__ import annotations

import uuid

from exaflow.rag.epilepsy.ingestion.models import ChunkRecord
from exaflow.rag.epilepsy.vectorstore.qdrant_schema import VECTOR_NAME


def build_qdrant_point(chunk: ChunkRecord, vector: list[float]) -> dict:
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
        "vector": {VECTOR_NAME: vector},
        "payload": chunk.metadata,
    }


def upsert_document_chunks(chunks, embedder, store) -> int:
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.embed_documents(texts)
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match chunk count")
    points = [
        build_qdrant_point(chunk, vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    store.ensure_collection()
    store.upsert_points(points)
    return len(points)
