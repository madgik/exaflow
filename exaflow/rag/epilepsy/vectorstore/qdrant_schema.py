"""Qdrant schema constants for epilepsy RAG."""

from __future__ import annotations

COLLECTION_NAME = "epilepsy_rag_v1"
VECTOR_NAME = "medcpt_dense"
VECTOR_DISTANCE = "Cosine"
DEFAULT_TOP_K = 8

DEFAULT_FILTERS = {
    "review_status": "approved",
    "clinical_domain": "epilepsy",
}

PAYLOAD_INDEX_FIELDS = (
    ("chunk_id", "keyword"),
    ("document_id", "keyword"),
    ("source_type", "keyword"),
    ("organization", "keyword"),
    ("citation_label", "keyword"),
    ("publication_year", "integer"),
    ("version", "keyword"),
    ("language", "keyword"),
    ("jurisdiction", "keyword"),
    ("patient_group", "keyword"),
    ("clinical_domain", "keyword"),
    ("review_status", "keyword"),
    ("evidence_level", "keyword"),
    ("trust_policy", "keyword"),
)
