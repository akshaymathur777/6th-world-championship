"""Freeze a section-aligned question set before any ablation.

chymaabs/CDCQA was not on the Hugging Face Hub (checked 2026-11-17), so questions
are written once from CDC sections and then locked. Later edits fail the scorer.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from tqdm import tqdm

from src.ingest import load_corpus
from src.ollama import chat
from src.paths import CORPUS_MANIFEST, FREEZE, QUESTIONS
from src.textutil import evidence_quote, overlaps_heading, split_sections, word_count

TARGET = 200
MIN_WORDS = 80
MAX_WORDS = 400
QUESTION_MODEL = "llama3.2:3b"
PARTIAL = QUESTIONS.with_suffix(".partial.jsonl")

SYSTEM = (
    "You write one clinical question that a physician could ask at the point of care. "
    "The question must be answerable from the passage. "
    "Do not copy the section heading. Do not quote the passage. "
    "Return only the question, one sentence, with no preamble."
)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(doc_id: str, section: str) -> str:
    return hashlib.sha256(f"{doc_id}\n{section}".encode("utf-8")).hexdigest()


def _candidates() -> list[dict]:
    rows = []
    for doc in load_corpus():
        for section in split_sections(doc["text"]):
            count = word_count(section.text)
            if count < MIN_WORDS or count > MAX_WORDS:
                continue
            if section.title.lower() in {"document", "table of contents", "references"}:
                continue
            rows.append(
                {
                    "key": _stable_key(doc["doc_id"], section.title),
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "section": section.title,
                    "text": section.text,
                }
            )
    rows.sort(key=lambda row: row["key"])
    return rows


def _clean_question(raw: str) -> str:
    line = ""
    for candidate in raw.strip().splitlines():
        candidate = candidate.strip().strip('"').strip()
        if candidate:
            line = candidate
            break
    line = line.removeprefix("Question:").strip()
    if not line.endswith("?"):
        line = line.rstrip(".") + "?"
    return line


def _acceptable(question: str, section: str) -> bool:
    if not question or overlaps_heading(question, section):
        return False
    words = question.split()
    if not 6 <= len(words) <= 50:
        return False
    lowered = question.lower()
    if "as an ai" in lowered or "language model" in lowered:
        return False
    return True


def _ask(row: dict) -> str | None:
    passage = row["text"][:3500]
    user = (
        f"Document title: {row['title']}\n"
        f"Section heading: {row['section']}\n"
        f"Passage:\n{passage}"
    )
    question = _clean_question(chat(QUESTION_MODEL, SYSTEM, user))
    if _acceptable(question, row["section"]):
        return question
    retry = _clean_question(
        chat(
            QUESTION_MODEL,
            SYSTEM,
            user + "\n\nThe previous draft copied the heading or was unusable. Write a different question.",
        )
    )
    if _acceptable(retry, row["section"]):
        return retry
    return None


def _load_partial() -> list[dict]:
    if not PARTIAL.exists():
        return []
    rows = []
    with PARTIAL.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def freeze_eval(force: bool = False) -> dict:
    if FREEZE.exists() and not force:
        print(f"Question set already frozen at {QUESTIONS}")
        return json.loads(FREEZE.read_text(encoding="utf-8"))
    if force and FREEZE.exists():
        raise RuntimeError("Refusing to overwrite a frozen question set. Delete eval/FREEZE.json only if you intend to discard every result.")

    QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    done = _load_partial()
    seen = {(row["doc_id"], row["section"]) for row in done}
    candidates = _candidates()
    print(f"{len(candidates)} eligible sections. {len(done)} questions already drafted.")
    with PARTIAL.open("a", encoding="utf-8") as handle:
        for row in tqdm(candidates, desc="draft questions"):
            if len(done) >= TARGET:
                break
            if (row["doc_id"], row["section"]) in seen:
                continue
            question = _ask(row)
            if not question:
                continue
            record = {
                "question_id": f"q{len(done) + 1:04d}",
                "doc_id": row["doc_id"],
                "title": row["title"],
                "url": row["url"],
                "section": row["section"],
                "question": question,
                "evidence_quote": evidence_quote(row["text"]),
                "demo": len(done) < 3,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done.append(record)
            seen.add((row["doc_id"], row["section"]))

    if len(done) < 50:
        raise RuntimeError(f"Only drafted {len(done)} questions. Need at least 50 before freezing.")

    with QUESTIONS.open("w", encoding="utf-8") as handle:
        for record in done:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question_model": QUESTION_MODEL,
        "target": TARGET,
        "section_words": [MIN_WORDS, MAX_WORDS],
        "n_questions": len(done),
        "n_demo": min(3, len(done)),
        "corpus_sha256": manifest["corpus_sha256"],
        "questions_sha256": _sha256(QUESTIONS),
        "cdcqa": "not on Hugging Face Hub on 2026-11-17; section questions used instead",
        "selection": "Eligible sections sorted by sha256(doc_id, section). Walk in that order until 200 accepted questions. First three are the demo queries.",
    }
    FREEZE.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(f"Froze {len(done)} questions.")
    return freeze
