"""Text cleaning helpers for epilepsy RAG ingestion."""

from __future__ import annotations

import re

_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTIBLANK_RE = re.compile(r"\n{3,}")
_HYPHENATED_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def remove_page_number_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not _PAGE_NUMBER_RE.match(line)
    )


def fix_hyphenation(text: str) -> str:
    return _HYPHENATED_BREAK_RE.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    lines = [_MULTISPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    normalized = "\n".join(line for line in lines if line)
    return _MULTIBLANK_RE.sub("\n\n", normalized).strip()


def clean_text(
    text: str,
    *,
    remove_page_numbers: bool = True,
    normalize_space: bool = True,
    fix_hyphenated_breaks: bool = True,
) -> str:
    cleaned = text
    if remove_page_numbers:
        cleaned = remove_page_number_lines(cleaned)
    if fix_hyphenated_breaks:
        cleaned = fix_hyphenation(cleaned)
    if normalize_space:
        cleaned = normalize_whitespace(cleaned)
    return cleaned
