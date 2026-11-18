"""Download the redistributable CDC slice and write a local JSONL corpus."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from tqdm import tqdm

from src.paths import CORPUS, CORPUS_DIR, CORPUS_MANIFEST, DROPPED
from src.textutil import document_title, first_cdc_url, third_party_notice, word_count


def _sha256_file(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest(force: bool = False) -> dict:
    if CORPUS.exists() and CORPUS_MANIFEST.exists() and not force:
        return json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))

    from datasets import load_dataset

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("epfl-llm/guidelines", split="train", streaming=True)
    kept = 0
    dropped = 0
    words = 0
    with_url = 0
    CORPUS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS.open("w", encoding="utf-8") as corpus, DROPPED.open("w", encoding="utf-8") as rejects:
        for row in tqdm(dataset, desc="scan guidelines", unit="doc"):
            if (row.get("source") or "").lower() != "cdc":
                continue
            clean = row.get("clean_text") or ""
            raw = row.get("raw_text") or ""
            reason = third_party_notice(clean) or third_party_notice(raw)
            doc_id = str(row.get("id") or "")
            if reason or word_count(clean) < 80 or not doc_id:
                dropped += 1
                rejects.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "reason": reason or "too short or missing id",
                        }
                    )
                    + "\n"
                )
                continue
            url = first_cdc_url(raw) or first_cdc_url(clean)
            record = {
                "doc_id": doc_id,
                "title": document_title(clean),
                "url": url or f"cdc:{doc_id}",
                "url_recovered": bool(url),
                "text": clean,
            }
            kept += 1
            words += word_count(clean)
            with_url += int(bool(url))
            corpus.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source_dataset": "epfl-llm/guidelines",
        "source_tag": "cdc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "documents": kept,
        "dropped": dropped,
        "words": words,
        "documents_with_cdc_url": with_url,
        "corpus_sha256": _sha256_file(CORPUS),
        "note": (
            "Only source=cdc rows are kept. The upstream release leaves CDC title and "
            "url empty, so title is recovered from headings and url from the first cdc.gov "
            "link in the raw text. Documents with a third-party notice in the opening are dropped."
        ),
    }
    CORPUS_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Kept {kept} CDC documents ({words} words, {with_url} with a cdc.gov url). "
        f"Dropped {dropped}."
    )
    return manifest


def load_corpus() -> list[dict]:
    if not CORPUS.exists():
        raise FileNotFoundError("Corpus missing. Run: python -m src ingest")
    documents = []
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(json.loads(line))
    return documents
