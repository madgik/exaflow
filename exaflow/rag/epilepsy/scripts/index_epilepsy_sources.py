"""Index approved epilepsy sources into Qdrant."""

from __future__ import annotations

import argparse
from pathlib import Path

from exaflow.rag.epilepsy.embeddings.medcpt import build_embedder
from exaflow.rag.epilepsy.ingestion.pipeline import run_ingestion_pipeline
from exaflow.rag.epilepsy.vectorstore.indexer import upsert_document_chunks
from exaflow.rag.epilepsy.vectorstore.qdrant_client import QdrantConfig
from exaflow.rag.epilepsy.vectorstore.qdrant_client import QdrantEpilepsyStore


def _default_sources_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "sources.yaml"


def _default_chunking_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "chunking.yaml"


def _default_qdrant_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "qdrant"


def main() -> int:
    parser = argparse.ArgumentParser(description="Index epilepsy sources into Qdrant.")
    parser.add_argument("--sources", default=str(_default_sources_path()))
    parser.add_argument("--chunking", default=str(_default_chunking_path()))
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-path", default=str(_default_qdrant_path()))
    parser.add_argument("--remote-qdrant", action="store_true")
    parser.add_argument("--embedder", choices=["auto", "medcpt", "hash"], default="auto")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    chunks, report = run_ingestion_pipeline(args.sources, args.chunking)
    if not chunks:
        raise SystemExit("No chunks were produced. Check source files and policies.")

    embedder = build_embedder(args.embedder, device=args.device)
    store = QdrantEpilepsyStore(
        QdrantConfig(
            url=args.qdrant_url,
            path=None if args.remote_qdrant else args.qdrant_path,
            embedding_size=embedder.embedding_size(),
        )
    )
    indexed_count = upsert_document_chunks(chunks, embedder, store)

    print(
        f"[INDEX] documents={len(report.documents)} chunks={indexed_count} embedder={args.embedder}"
    )
    for item in report.documents:
        print(
            f"[DOC] id={item.document_id} sections={item.section_count} chunks={item.chunk_count}"
        )
    print(f"[QDRANT] collection_points={store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
