"""Readiness checks for the epilepsy RAG MVP."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _check_dependency(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    required_files = [
        raw_dir / "ilae_2025_seizure_classification.html",
        raw_dir / "ilae_epilepsy_syndromes.html",
    ]
    print("[FILES]")
    for file_path in required_files:
        print(f"{file_path.name}: {'OK' if file_path.exists() else 'MISSING'}")

    print("[DEPS]")
    for module_name in ("yaml", "qdrant_client", "transformers", "torch"):
        print(f"{module_name}: {'OK' if _check_dependency(module_name) else 'MISSING'}")

    qdrant_path = root / "data" / "qdrant"
    print("[LOCAL_QDRANT]")
    print(f"path={qdrant_path}")
    print(f"exists={'YES' if qdrant_path.exists() else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
