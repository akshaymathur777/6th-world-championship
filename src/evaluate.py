"""The only script that reports canonical Recall@5."""

from __future__ import annotations

import json
import re
import time

import numpy as np

from src.embed import embed_query
from src.index import directory_size, index_dir, open_collection
from src.paths import CORPUS_MANIFEST, FREEZE, QUESTIONS


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def is_hit(document: str, metadata: dict, question: dict) -> tuple[bool, bool]:
    """Return (section_match, quote_match). A Recall@5 hit is either one."""
    section_match = (
        (metadata or {}).get("url") == question["url"]
        and (metadata or {}).get("section") == question["section"]
    )
    quote = _norm(question.get("evidence_quote") or "")
    quote_match = len(quote) >= 20 and quote in _norm(document or "")
    return section_match, quote_match


def load_questions() -> list[dict]:
    _require_freeze()
    questions = []
    with QUESTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def _require_freeze() -> None:
    if not FREEZE.exists() or not QUESTIONS.exists() or not CORPUS_MANIFEST.exists():
        raise FileNotFoundError("Eval set is not frozen. Run: python -m src freeze-eval")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    if freeze.get("corpus_sha256") != manifest.get("corpus_sha256"):
        raise RuntimeError("Corpus changed after the question set was frozen. Refusing to score.")
    if freeze.get("questions_sha256") != _sha256(QUESTIONS):
        raise RuntimeError("Question file changed after freeze. Refusing to score.")


def _sha256(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_se(hits: np.ndarray, draws: int = 1000) -> float:
    if len(hits) == 0:
        return 0.0
    rng = np.random.default_rng(0)
    means = np.empty(draws, dtype=np.float64)
    n = len(hits)
    for i in range(draws):
        means[i] = hits[rng.integers(0, n, n)].mean()
    return float(means.std(ddof=1))


def evaluate(run_id: str, embedding_model: str) -> dict:
    questions = load_questions()
    client, collection = open_collection(run_id)
    try:
        latencies = []
        section_hits = []
        quote_hits = []
        combined = []
        for question in questions:
            started = time.perf_counter()
            vector = embed_query(question["question"], embedding_model)
            result = collection.query(
                query_embeddings=[vector],
                n_results=5,
                include=["documents", "metadatas"],
            )
            latencies.append((time.perf_counter() - started) * 1000)
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            section_hit = False
            quote_hit = False
            for document, metadata in zip(documents, metadatas):
                section_match, quote_match = is_hit(document, metadata or {}, question)
                section_hit = section_hit or section_match
                quote_hit = quote_hit or quote_match
            section_hits.append(section_hit)
            quote_hits.append(quote_hit)
            combined.append(section_hit or quote_hit)
    finally:
        client.close()

    combined_arr = np.asarray(combined, dtype=np.float64)
    recall = float(combined_arr.mean()) if len(combined_arr) else 0.0
    se = bootstrap_se(combined_arr)
    latencies_arr = np.asarray(latencies, dtype=np.float64)
    p50 = float(np.percentile(latencies_arr, 50)) if len(latencies_arr) else 0.0
    report = {
        "n_questions": len(questions),
        "recall_at_5": recall,
        "recall_se": se,
        "noise_floor": 2 * se,
        "section_hit_rate": float(np.mean(section_hits)) if section_hits else 0.0,
        "quote_hit_rate": float(np.mean(quote_hits)) if quote_hits else 0.0,
        "p50_latency_ms": p50,
        "index_bytes": directory_size(index_dir(run_id)),
    }
    print(
        "CANONICAL Recall@5 = "
        f"{report['recall_at_5']:.4f} "
        f"(n={report['n_questions']}, se={report['recall_se']:.4f}, "
        f"noise_floor={report['noise_floor']:.4f})"
    )
    return report
