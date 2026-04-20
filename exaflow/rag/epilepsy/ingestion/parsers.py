"""HTML-first parsing helpers for epilepsy RAG ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser

from exaflow.rag.epilepsy.ingestion.cleaners import clean_text
from exaflow.rag.epilepsy.ingestion.models import ParsedDocument
from exaflow.rag.epilepsy.ingestion.models import ParsedSection
from exaflow.rag.epilepsy.ingestion.models import RawDocument
from exaflow.rag.epilepsy.ingestion.models import SourceConfig


@dataclass
class _SectionBuffer:
    level: int
    title: str
    text_parts: list[str]
    path_titles: list[str]


class _HeadingHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_text = ""
        self._capture_title = False
        self._capture_heading = False
        self._heading_level = 0
        self._heading_text_parts: list[str] = []
        self._current_text_parts: list[str] = []
        self._sections: list[_SectionBuffer] = []
        self._heading_stack: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._capture_title = True
            return
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self._flush_pending_text()
            self._capture_heading = True
            self._heading_level = int(tag[1])
            self._heading_text_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._capture_title = False
            return
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit() and self._capture_heading:
            heading_title = clean_text("".join(self._heading_text_parts))
            if heading_title:
                self._open_section(self._heading_level, heading_title)
            self._capture_heading = False
            self._heading_level = 0
            self._heading_text_parts = []

    def handle_data(self, data: str) -> None:
        text = unescape(data)
        if self._capture_title:
            self.title_text += text
            return
        if self._capture_heading:
            self._heading_text_parts.append(text)
            return
        self._current_text_parts.append(text)

    def parsed_sections(self) -> list[ParsedSection]:
        self._flush_pending_text()
        parsed: list[ParsedSection] = []
        for ordinal, section in enumerate(self._sections, start=1):
            section_text = clean_text(" ".join(section.text_parts))
            if not section_text:
                continue
            parsed.append(
                ParsedSection(
                    ordinal=ordinal,
                    level=section.level,
                    title=section.title,
                    path=" > ".join(section.path_titles),
                    text=section_text,
                )
            )
        return parsed

    def _open_section(self, level: int, title: str) -> None:
        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()
        self._heading_stack.append((level, title))
        self._sections.append(
            _SectionBuffer(
                level=level,
                title=title,
                text_parts=[],
                path_titles=[item[1] for item in self._heading_stack],
            )
        )

    def _flush_pending_text(self) -> None:
        text = clean_text(" ".join(self._current_text_parts))
        self._current_text_parts = []
        if not text:
            return
        if not self._sections:
            self._open_section(level=1, title="Document")
        self._sections[-1].text_parts.append(text)


def parse_html_document(raw_document: RawDocument, source: SourceConfig) -> ParsedDocument:
    html_text = raw_document.raw_bytes.decode("utf-8", errors="replace")
    parser = _HeadingHTMLParser()
    parser.feed(html_text)
    sections = parser.parsed_sections()
    cleaned_text = clean_text("\n\n".join(section.text for section in sections))
    return ParsedDocument(
        document_id=source.document_id,
        title=clean_text(parser.title_text) or source.title,
        raw_text=html_text,
        cleaned_text=cleaned_text,
        sections=sections,
        source_hash="",
    )


def parse_document(raw_document: RawDocument, source: SourceConfig) -> ParsedDocument:
    if raw_document.media_type != "text/html":
        raise ValueError(
            f"Only HTML ingestion is implemented in MVP, got '{raw_document.media_type}'"
        )
    return parse_html_document(raw_document, source)
