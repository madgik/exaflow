"""Metadata enrichment and synonym extraction for epilepsy RAG."""

from __future__ import annotations

from collections.abc import Iterable


def _extract_alias_hits(text: str, alias_map: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    hits: set[str] = set()
    for canonical, aliases in alias_map.items():
        candidates = [canonical, *aliases]
        if any(candidate.lower() in lowered for candidate in candidates):
            hits.add(canonical)
            hits.update(alias for alias in aliases if alias.lower() in lowered)
    return sorted(hits)


def build_synonym_metadata(text: str, normalization_cfg: dict) -> dict[str, list[str]]:
    drug_aliases = _extract_alias_hits(text, normalization_cfg.get("drug_alias_map", {}))
    syndrome_aliases = _extract_alias_hits(
        text, normalization_cfg.get("syndrome_alias_map", {})
    )
    return {
        "synonyms": sorted(set(drug_aliases + syndrome_aliases)),
        "drug_entities": drug_aliases,
        "disease_entities": syndrome_aliases,
    }


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
