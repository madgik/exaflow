"""Retriever for epilepsy RAG."""

from __future__ import annotations

from exaflow.rag.epilepsy.ingestion.models import RetrievedChunk
from exaflow.rag.epilepsy.vectorstore.qdrant_schema import DEFAULT_FILTERS
from exaflow.rag.epilepsy.vectorstore.qdrant_schema import DEFAULT_TOP_K


def _merge_filters(
    *,
    patient_group: str | None = None,
    guideline_only: bool = False,
    extra_filters: dict | None = None,
) -> dict:
    filters = dict(DEFAULT_FILTERS)
    if patient_group:
        filters["patient_group"] = patient_group
    if guideline_only:
        filters["source_type"] = "guideline"
    if extra_filters:
        filters.update({k: v for k, v in extra_filters.items() if v is not None})
    return filters


def retrieve_epilepsy_context(
    *,
    query: str,
    embedder,
    store,
    top_k: int = DEFAULT_TOP_K,
    patient_group: str | None = None,
    guideline_only: bool = False,
    extra_filters: dict | None = None,
) -> list[RetrievedChunk]:
    query_vector = embedder.embed_query(query)
    filter_values = _merge_filters(
        patient_group=patient_group,
        guideline_only=guideline_only,
        extra_filters=extra_filters,
    )
    payload_filter = store.build_filter(filter_values)
    search_result = store.search(
        query_vector=query_vector,
        top_k=top_k,
        payload_filter=payload_filter,
    )

    points = getattr(search_result, "points", search_result)
    retrieved: list[RetrievedChunk] = []
    for point in points:
        payload = point.payload
        retrieved.append(
            RetrievedChunk(
                chunk_id=payload["chunk_id"],
                document_id=payload["document_id"],
                citation_label=payload["citation_label"],
                score=float(point.score),
                text=payload["text"],
                section_title=payload["section_title"],
                source_url=payload["source_url"],
                publication_year=int(payload["publication_year"]),
                evidence_level=payload["evidence_level"],
                metadata=payload,
            )
        )
    return retrieved
