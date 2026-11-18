"""Shared text parsing for ingest, chunking, and the frozen question set."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")
WORD_RE = re.compile(r"\S+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CDC_URL_RE = re.compile(r"https?://(?:www\.)?cdc\.gov[^\s<>\"')\]]+", re.IGNORECASE)
BOILERPLATE_HEADING = re.compile(
    r"^(table of contents|contents|list of (tables|figures|boxes)|"
    r"summary|background|introduction|references|bibliography|"
    r"acknowledg(e)?ments?|preface|foreword|abstract|appendix|abbreviations?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Section:
    index: int
    title: str
    text: str

    @property
    def words(self) -> list[str]:
        return WORD_RE.findall(self.text)


def normalize_text(text: str) -> str:
    """Make heading markers line-leading. Upstream cleaning sometimes glues them on."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(?<!\n)(#{1,6})(?=\s|[A-Za-z])", r"\n\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def split_sections(text: str, min_words: int = 40) -> list[Section]:
    """Split on markdown headings and merge tiny fragments into the previous section."""
    normalized = normalize_text(text)
    raw: list[tuple[str, list[str]]] = []
    title = "Document"
    buffer: list[str] = []
    for line in normalized.split("\n"):
        match = HEADING_RE.match(line.strip())
        if match and match.group(2):
            raw.append((title, buffer))
            title = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    raw.append((title, buffer))

    merged: list[tuple[str, str]] = []
    for heading, lines in raw:
        body = "\n".join(lines).strip()
        if not body:
            continue
        if word_count(body) < min_words and merged:
            prev_title, prev_body = merged[-1]
            merged[-1] = (prev_title, f"{prev_body}\n\n{heading}\n{body}".strip())
        else:
            merged.append((heading.strip() or "Document", body))

    return [
        Section(index=i, title=title[:300], text=body)
        for i, (title, body) in enumerate(merged)
    ]


def document_title(text: str) -> str:
    """Best-effort title. The CDC slice does not populate the dataset title field."""
    normalized = normalize_text(text)
    for sentence in SENTENCE_RE.split(normalized[:2500]):
        if not re.search(r"guideline|criteria document|recommendations for", sentence, re.I):
            continue
        clause = re.split(r"\b(provides|describes|presents|summarizes|is)\b", sentence, maxsplit=1)[0]
        words = WORD_RE.findall(clause)
        if 6 <= len(words) <= 28:
            return " ".join(words)[:300]
    headings = []
    for line in normalized.split("\n"):
        match = HEADING_RE.match(line.strip())
        if match and match.group(2):
            headings.append(match.group(2).strip())
    for heading in headings:
        if re.search(r"guideline|recommendation|compendium|criteria", heading, re.I):
            if 3 <= word_count(heading) <= 24:
                return heading[:300]
    for heading in headings:
        if BOILERPLATE_HEADING.match(heading):
            continue
        if 3 <= word_count(heading) <= 18:
            return heading[:300]
    first = SENTENCE_RE.split(normalized)[0].strip()
    words = WORD_RE.findall(first)
    if not words:
        return "CDC guidance"
    return " ".join(words[:18])[:300]


def first_cdc_url(raw_text: str) -> str:
    match = CDC_URL_RE.search(raw_text or "")
    if not match:
        return ""
    return match.group(0).rstrip(".,);>")


def third_party_notice(text: str) -> str | None:
    """Return a drop reason when the opening of a document is not CDC/NIOSH work.

    Notices are read from the first 4000 characters so a later discussion of
    copyright law does not discard the document.
    """
    window = (text or "")[:4000].lower()
    if "all rights reserved" in window:
        return "all rights reserved"
    if "reprinted with permission" in window:
        return "reprinted with permission"
    if "published as a courtesy" in window:
        return "published as a courtesy to a non-CDC organization"
    for match in re.finditer(
        r"copyright\s*(?:©|\(c\))?\s*((?:19|20)\d{2}.{0,80})",
        window,
    ):
        holder = match.group(0)
        if any(
            token in holder
            for token in (
                "cdc",
                "niosh",
                "dhhs",
                "hhs",
                "u.s.",
                "united states",
                "public domain",
                "government",
            )
        ):
            continue
        return "third-party copyright line"
    return None


def evidence_quote(text: str) -> str:
    """A sentence from the section, used as an alternate Recall@5 hit condition."""
    sentences = [part.strip() for part in SENTENCE_RE.split(text.strip()) if part.strip()]
    candidates = [sentence for sentence in sentences if 8 <= word_count(sentence) <= 60]
    if not candidates:
        words = WORD_RE.findall(text)
        return " ".join(words[:40])
    def rank(sentence: str) -> tuple[int, int]:
        distinctive = 1 if re.search(r"\d|recommend", sentence, re.I) else 0
        return (distinctive, word_count(sentence))

    return max(candidates, key=rank)


def overlaps_heading(question: str, heading: str) -> bool:
    """True when the question copies a long run of the section heading."""
    def grams(value: str, size: int) -> set[tuple[str, ...]]:
        tokens = re.findall(r"[a-z0-9]+", value.lower())
        if len(tokens) < size:
            return set()
        return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}

    size = 5 if word_count(heading) < 8 else 6
    return bool(grams(question, size) & grams(heading, size))
