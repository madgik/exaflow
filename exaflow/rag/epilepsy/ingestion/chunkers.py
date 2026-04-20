"""Section-aware chunking for epilepsy RAG ingestion."""

from __future__ import annotations

import hashlib
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from exaflow.rag.epilepsy.ingestion.models import ChunkRecord
from exaflow.rag.epilepsy.ingestion.models import ParsedDocument
from exaflow.rag.epilepsy.ingestion.models import ParsedSection
from exaflow.rag.epilepsy.ingestion.models import SourceConfig
from exaflow.rag.epilepsy.ingestion.normalizers import build_synonym_metadata


def _tokenize(text: str) -> list[str]:
    return text.split()


def _deterministic_chunk_id(document_id: str, section_ordinal: int, chunk_index: int) -> str:
    return f"{document_id}__sec{section_ordinal:03d}__chunk{chunk_index:03d}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_chunk_texts(section: ParsedSection, chunk_cfg: dict[str, Any]) -> list[str]:
    tokens = _tokenize(section.text)
    if not tokens:
        return []

    target = int(chunk_cfg["chunking"]["target_tokens"])
    overlap = int(chunk_cfg["chunking"]["overlap_tokens"])
    max_tokens = int(chunk_cfg["chunking"]["max_tokens"])

    window = min(target, max_tokens)
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + window, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _build_chunk_metadata(
    source: SourceConfig,
    section: ParsedSection,
    chunk_index: int,
    chunk_text: str,
    chunk_cfg: dict[str, Any],
) -> dict[str, Any]:
    normalization_cfg = chunk_cfg.get("normalization", {})
    synonym_data = build_synonym_metadata(chunk_text, normalization_cfg)
    return {
        "chunk_id": _deterministic_chunk_id(source.document_id, section.ordinal, chunk_index),
        "document_id": source.document_id,
        "source_type": source.source_type,
        "organization": source.organization,
        "title": source.title,
        "citation_label": source.citation_label,
        "publication_year": source.publication_year,
        "version": source.version,
        "language": source.language,
        "jurisdiction": source.jurisdiction,
        "patient_group": source.patient_group,
        "clinical_domain": "epilepsy",
        "clinical_topics": source.clinical_topics,
        "section_title": section.title,
        "section_path": section.path,
        "chunk_index": chunk_index,
        "token_count": len(_tokenize(chunk_text)),
        "source_url": source.source_url,
        "source_hash": _source_hash(source.local_path),
        "review_status": source.review_status,
        "evidence_level": source.evidence_level,
        "trust_policy": source.trust_policy,
        "tags": source.tags,
        "drug_entities": synonym_data["drug_entities"],
        "disease_entities": synonym_data["disease_entities"],
        "synonyms": synonym_data["synonyms"],
        "ingestion_timestamp": _utc_now(),
        "text": chunk_text,
    }


def build_chunks(
    parsed_document: ParsedDocument,
    source: SourceConfig,
    chunk_cfg: dict[str, Any],
) -> list[ChunkRecord]:
    quality_cfg = chunk_cfg.get("quality_rules", {})
    min_tokens = int(quality_cfg.get("reject_chunks_shorter_than_tokens", 0))

    chunks: list[ChunkRecord] = []
    seen_texts: set[str] = set()
    for section in parsed_document.sections:
        chunk_texts = _build_chunk_texts(section, chunk_cfg)
        section_chunks_before = len(chunks)
        for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
            token_count = len(_tokenize(chunk_text))
            if token_count < min_tokens:
                continue
            normalized_text = chunk_text.strip()
            if not normalized_text:
                continue
            if quality_cfg.get("reject_duplicate_chunks", False) and normalized_text in seen_texts:
                continue
            seen_texts.add(normalized_text)
            metadata = _build_chunk_metadata(
                source=source,
                section=section,
                chunk_index=chunk_index,
                chunk_text=normalized_text,
                chunk_cfg=chunk_cfg,
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=metadata["chunk_id"],
                    document_id=source.document_id,
                    text=normalized_text,
                    section_title=section.title,
                    section_path=section.path,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    metadata=metadata,
                )
            )
        if len(chunks) == section_chunks_before:
            fallback_text = section.text.strip()
            fallback_tokens = len(_tokenize(fallback_text))
            if fallback_text and fallback_text not in seen_texts:
                metadata = _build_chunk_metadata(
                    source=source,
                    section=section,
                    chunk_index=1,
                    chunk_text=fallback_text,
                    chunk_cfg=chunk_cfg,
                )
                seen_texts.add(fallback_text)
                chunks.append(
                    ChunkRecord(
                        chunk_id=metadata["chunk_id"],
                        document_id=source.document_id,
                        text=fallback_text,
                        section_title=section.title,
                        section_path=section.path,
                        chunk_index=1,
                        token_count=fallback_tokens,
                        metadata=metadata,
                    )
                )
    return chunks
