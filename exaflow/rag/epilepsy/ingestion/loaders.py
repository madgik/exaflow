"""Load source registry and raw source documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from exaflow.rag.epilepsy.ingestion.models import RawDocument
from exaflow.rag.epilepsy.ingestion.models import SourceConfig


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at YAML root in '{path}'")
    return loaded


def load_sources_registry(path: str | Path) -> list[SourceConfig]:
    payload = load_yaml_file(path)
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("sources registry must contain a 'sources' list")
    return [SourceConfig(**source) for source in raw_sources]


def load_chunking_policy(path: str | Path) -> dict[str, Any]:
    return load_yaml_file(path)


def select_approved_sources(sources: list[SourceConfig]) -> list[SourceConfig]:
    selected = [
        source
        for source in sources
        if source.enabled and source.approved_for_rag and source.review_status == "approved"
    ]
    return sorted(selected, key=lambda source: (source.priority, source.document_id))


def detect_media_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".html", ".htm"}:
        return "text/html"
    if suffix in {".txt", ".md"}:
        return "text/plain"
    if suffix == ".pdf":
        return "application/pdf"
    raise ValueError(f"Unsupported source media type for '{path}'")


def load_raw_source(source: SourceConfig) -> RawDocument:
    path = Path(source.local_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Source '{source.document_id}' raw file not found: '{source.local_path}'"
        )
    raw_bytes = path.read_bytes()
    if not raw_bytes:
        raise ValueError(f"Source '{source.document_id}' is empty: '{source.local_path}'")
    return RawDocument(
        document_id=source.document_id,
        source_type=source.source_type,
        media_type=detect_media_type(path),
        local_path=str(path),
        raw_bytes=raw_bytes,
    )
