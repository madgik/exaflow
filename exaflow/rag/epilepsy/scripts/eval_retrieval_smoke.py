"""Run a small retrieval smoke evaluation against the epilepsy eval set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exaflow.rag.epilepsy.embeddings.medcpt import build_embedder
from exaflow.rag.epilepsy.vectorstore.qdrant_client import QdrantConfig
from exaflow.rag.epilepsy.vectorstore.qdrant_client import QdrantEpilepsyStore
from exaflow.rag.epilepsy.vectorstore.retriever import retrieve_epilepsy_context


def _default_eval_path() -> Path:
    return Path(__file__).resolve().parents[1] / "evaluation" / "epilepsy_eval_v1.jsonl"


def _default_qdrant_path() -> str:
    return str((Path(__file__).resolve().parents[1] / "data" / "qdrant"))


def _load_cases(path: str | Path, limit: int) -> list[dict]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(payload)
        if len(cases) >= limit:
            break
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval smoke evaluation.")
    parser.add_argument("--eval-path", default=str(_default_eval_path()))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-path", default=_default_qdrant_path())
    parser.add_argument("--remote-qdrant", action="store_true")
    parser.add_argument("--embedder", choices=["auto", "medcpt", "hash"], default="auto")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cases = _load_cases(args.eval_path, args.limit)
    embedder = build_embedder(args.embedder, device=args.device)
    store = QdrantEpilepsyStore(
        QdrantConfig(
            url=args.qdrant_url,
            path=None if args.remote_qdrant else args.qdrant_path,
            embedding_size=embedder.embedding_size(),
        )
    )

    hits = 0
    for case in cases:
        retrieved = retrieve_epilepsy_context(
            query=case["question"],
            embedder=embedder,
            store=store,
            top_k=args.top_k,
        )
        retrieved_sources = {item.document_id for item in retrieved}
        expected_sources = set(case["expected_sources"])
        matched = not expected_sources or bool(retrieved_sources & expected_sources)
        hits += int(matched)
        print(
            f"[EVAL] id={case['question_id']} matched={matched} expected={sorted(expected_sources)} retrieved={sorted(retrieved_sources)}"
        )

    print(f"[SUMMARY] matched={hits}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
