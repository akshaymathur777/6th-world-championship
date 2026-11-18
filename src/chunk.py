"""Chunkers. Every chunk carries url, title, section, and chunk_id."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.textutil import WORD_RE, Section, split_sections, word_count

REC_BREAK = re.compile(
    r"^(recommendation\b|recommended\b|we recommend\b|the cdc recommends\b|"
    r"cdc recommends\b|dosage\b|dose\b|dosing\b|treatment\b|regimen\b|\d+\.\s+\S)",
    re.IGNORECASE,
)
DOSE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|units|iu)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    section: str
    text: str

    def embedded_document(self) -> str:
        return f"Title: {self.title}\nSection: {self.section}\n{self.text}"

    def metadata(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_reference_id": self.doc_id,
            "file_name": f"{self.doc_id}.md",
            "title": self.title[:300],
            "url": self.url[:500],
            "section": self.section[:300],
        }


def _window_spans(n_words: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= overlap:
        raise ValueError("chunk size must exceed overlap")
    if n_words == 0:
        return []
    step = size - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_words:
        end = min(n_words, start + size)
        if end - start < 20 and spans:
            spans[-1] = (spans[-1][0], end)
            break
        spans.append((start, end))
        if end == n_words:
            break
        start += step
    return spans


def _from_section_words(
    doc_id: str,
    title: str,
    url: str,
    section: Section,
    size: int,
    overlap: int,
) -> list[Chunk]:
    words = section.words
    spans = (
        _window_spans(len(words), size, overlap)
        if len(words) > size
        else [(0, len(words))]
    )
    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(spans):
        body = " ".join(words[start:end]).strip()
        if word_count(body) < 8:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{section.index}:{index}",
                doc_id=doc_id,
                title=title,
                url=url,
                section=section.title,
                text=body,
            )
        )
    return chunks


def chunk_fixed(doc: dict, size: int, overlap: int) -> list[Chunk]:
    """Whitespace-word windows. Section metadata is the section owning the most words."""
    sections = split_sections(doc["text"])
    labeled: list[tuple[str, str]] = []
    for section in sections:
        for word in section.words:
            labeled.append((word, section.title))
    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(_window_spans(len(labeled), size, overlap)):
        span = labeled[start:end]
        counts: dict[str, int] = {}
        for _, section_title in span:
            counts[section_title] = counts.get(section_title, 0) + 1
        section_title = max(counts, key=lambda key: counts[key]) if counts else "Document"
        body = " ".join(word for word, _ in span).strip()
        if word_count(body) < 8:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc['doc_id']}:fixed:{index}",
                doc_id=doc["doc_id"],
                title=doc["title"],
                url=doc["url"],
                section=section_title,
                text=body,
            )
        )
    return chunks


def chunk_section(doc: dict, size: int = 256, overlap: int = 40) -> list[Chunk]:
    """One chunk per heading. Sections longer than `size` words are windowed."""
    chunks: list[Chunk] = []
    for section in split_sections(doc["text"]):
        chunks.extend(
            _from_section_words(doc["doc_id"], doc["title"], doc["url"], section, size, overlap)
        )
    return chunks


def _recommendation_pieces(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        return [text.strip()] if text.strip() else []
    pieces: list[list[str]] = []
    current: list[str] = []
    for paragraph in paragraphs:
        head = paragraph[:120]
        breaks = bool(REC_BREAK.match(head) or DOSE.search(head))
        if current and breaks:
            pieces.append(current)
            current = [paragraph]
        else:
            current.append(paragraph)
    if current:
        pieces.append(current)
    return ["\n\n".join(part) for part in pieces if part]


def chunk_recommendation(doc: dict, size: int = 256, overlap: int = 40) -> list[Chunk]:
    """Split sections on recommendation and dosing paragraphs, then window long pieces."""
    chunks: list[Chunk] = []
    for section in split_sections(doc["text"]):
        for piece_index, piece in enumerate(_recommendation_pieces(section.text)):
            words = WORD_RE.findall(piece)
            spans = (
                _window_spans(len(words), size, overlap)
                if len(words) > size
                else [(0, len(words))]
            )
            for index, (start, end) in enumerate(spans):
                body = " ".join(words[start:end]).strip()
                if word_count(body) < 8:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc['doc_id']}:{section.index}:{piece_index}:{index}",
                        doc_id=doc["doc_id"],
                        title=doc["title"],
                        url=doc["url"],
                        section=section.title,
                        text=body,
                    )
                )
    return chunks


CHUNKERS = {
    "fixed_128": lambda doc: chunk_fixed(doc, 128, 20),
    "fixed_256": lambda doc: chunk_fixed(doc, 256, 40),
    "section": lambda doc: chunk_section(doc, 256, 40),
    "recommendation": lambda doc: chunk_recommendation(doc, 256, 40),
}


def chunk_document(doc: dict, chunker: str) -> list[Chunk]:
    if chunker not in CHUNKERS:
        raise KeyError(f"Unknown chunker {chunker}. Choices: {sorted(CHUNKERS)}")
    return CHUNKERS[chunker](doc)
