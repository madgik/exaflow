"""End-to-end ingestion orchestration for epilepsy RAG MVP."""

from __future__ import annotations

import hashlib
from pathlib import Path

from exaflow.rag.epilepsy.ingestion.chunkers import build_chunks
from exaflow.rag.epilepsy.ingestion.loaders import load_chunking_policy
from exaflow.rag.epilepsy.ingestion.loaders import load_raw_source
from exaflow.rag.epilepsy.ingestion.loaders import load_sources_registry
from exaflow.rag.epilepsy.ingestion.loaders import select_approved_sources
from exaflow.rag.epilepsy.ingestion.models import ChunkRecord
from exaflow.rag.epilepsy.ingestion.models import IngestionReport
from exaflow.rag.epilepsy.ingestion.parsers import parse_document


def _with_source_hash(parsed_document, local_path: str):
    source_hash = hashlib.sha256(Path(local_path).read_bytes()).hexdigest()
    return parsed_document.__class__(
        document_id=parsed_document.document_id,
        title=parsed_document.title,
        raw_text=parsed_document.raw_text,
        cleaned_text=parsed_document.cleaned_text,
        sections=parsed_document.sections,
        source_hash=source_hash,
    )


def ingest_source(source, chunking_policy) -> list[ChunkRecord]:
    raw_document = load_raw_source(source)
    parsed_document = parse_document(raw_document, source)
    parsed_document = _with_source_hash(parsed_document, source.local_path)
    return build_chunks(parsed_document, source, chunking_policy)


def run_ingestion_pipeline(
    source_registry_path: str | Path,
    chunking_policy_path: str | Path,
) -> tuple[list[ChunkRecord], IngestionReport]:
    sources = load_sources_registry(source_registry_path)
    approved_sources = select_approved_sources(sources)
    chunking_policy = load_chunking_policy(chunking_policy_path)

    all_chunks: list[ChunkRecord] = []
    report = IngestionReport()
    for source in approved_sources:
        chunks = ingest_source(source, chunking_policy)
        all_chunks.extend(chunks)
        section_paths = {chunk.section_path for chunk in chunks}
        report.add(
            document_id=source.document_id,
            chunk_count=len(chunks),
            section_count=len(section_paths),
        )
    return all_chunks, report
