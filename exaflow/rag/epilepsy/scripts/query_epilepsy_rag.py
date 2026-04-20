"""Query the epilepsy RAG retriever."""

from __future__ import annotations

import argparse
from pathlib import Path

from exaflow.rag.epilepsy.embeddings.medcpt import build_embedder
from exaflow.rag.epilepsy.vectorstore.qdrant_client import QdrantConfig
from exaflow.rag.epilepsy.vectorstore.qdrant_client import QdrantEpilepsyStore
from exaflow.rag.epilepsy.vectorstore.retriever import retrieve_epilepsy_context


def _default_qdrant_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "data" / "qdrant")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the epilepsy RAG retriever.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-path", default=_default_qdrant_path())
    parser.add_argument("--remote-qdrant", action="store_true")
    parser.add_argument("--guideline-only", action="store_true")
    parser.add_argument("--patient-group")
    parser.add_argument("--embedder", choices=["auto", "medcpt", "hash"], default="auto")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    embedder = build_embedder(args.embedder, device=args.device)
    store = QdrantEpilepsyStore(
        QdrantConfig(
            url=args.qdrant_url,
            path=None if args.remote_qdrant else args.qdrant_path,
            embedding_size=embedder.embedding_size(),
        )
    )
    retrieved = retrieve_epilepsy_context(
        query=args.query,
        embedder=embedder,
        store=store,
        top_k=args.top_k,
        patient_group=args.patient_group,
        guideline_only=args.guideline_only,
    )

    print(f"[QUERY] {args.query}")
    for idx, item in enumerate(retrieved, start=1):
        print(
            f"[{idx}] score={item.score:.4f} source={item.citation_label} year={item.publication_year} section={item.section_title}"
        )
        print(item.text[:400])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
