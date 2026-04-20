"""Canonical models for epilepsy RAG ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    document_id: str
    enabled: bool
    approved_for_rag: bool
    priority: int
    source_type: str
    organization: str
    title: str
    version: str
    publication_year: int
    language: str
    jurisdiction: str
    patient_group: str
    clinical_topics: list[str]
    source_url: str
    acquisition_method: str
    local_path: str
    citation_label: str
    review_status: str
    evidence_level: str
    trust_policy: str
    synonym_policy: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RawDocument:
    document_id: str
    source_type: str
    media_type: str
    local_path: str
    raw_bytes: bytes


@dataclass(frozen=True)
class ParsedSection:
    ordinal: int
    level: int
    title: str
    path: str
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    title: str
    raw_text: str
    cleaned_text: str
    sections: list[ParsedSection]
    source_hash: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    text: str
    section_title: str
    section_path: str
    chunk_index: int
    token_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    citation_label: str
    score: float
    text: str
    section_title: str
    source_url: str
    publication_year: int
    evidence_level: str
    metadata: dict[str, Any]


@dataclass
class DocumentIngestionResult:
    document_id: str
    chunk_count: int
    section_count: int


@dataclass
class IngestionReport:
    documents: list[DocumentIngestionResult] = field(default_factory=list)

    def add(self, document_id: str, chunk_count: int, section_count: int) -> None:
        self.documents.append(
            DocumentIngestionResult(
                document_id=document_id,
                chunk_count=chunk_count,
                section_count=section_count,
            )
        )
