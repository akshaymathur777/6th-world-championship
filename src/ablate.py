"""Timeboxed ablations. A family stops when its challengers do not clear the noise floor."""

from __future__ import annotations

import json
import shutil
import time

from src.chunk import chunk_document
from src.embed import embed_chunks
from src.evaluate import evaluate
from src.experiments import append_row, beats, load_rows, update_decision
from src.index import build_index, index_dir
from src.ingest import load_corpus
from src.paths import WINNER

DEFAULT_M = 16
DEFAULT_CONSTRUCTION = 100
DEFAULT_SEARCH = 100
INCUMBENT_CHUNKER = "fixed_256"
INCUMBENT_MODEL = "nomic-embed-text"
EMBED_MODELS = ["nomic-embed-text", "all-minilm", "mxbai-embed-large"]
CHUNKERS = ["fixed_256", "fixed_128", "section", "recommendation"]
HNSW = [
    (DEFAULT_M, DEFAULT_CONSTRUCTION, DEFAULT_SEARCH),
    (32, 128, 64),
    (16, 200, 200),
]


def _run_id(chunker: str, model: str, m: int, construction: int, search: int) -> str:
    safe_model = model.replace(":", "-").replace("/", "-")
    return f"{chunker}__{safe_model}__m{m}c{construction}s{search}"


def _available_models() -> list[str]:
    import httpx

    from src.ollama import HOST

    response = httpx.get(f"{HOST}/api/tags", timeout=30)
    response.raise_for_status()
    names = set()
    for item in response.json().get("models") or []:
        name = item.get("name") or ""
        names.add(name.split(":")[0] if name.endswith(":latest") else name)
        names.add(name)
    usable = []
    for model in EMBED_MODELS:
        if model in names or f"{model}:latest" in names:
            usable.append(model)
    if INCUMBENT_MODEL not in usable:
        raise RuntimeError("nomic-embed-text is not available in Ollama.")
    return usable


def _chunks_for(documents: list[dict], chunker: str):
    chunks = []
    for doc in documents:
        chunks.extend(chunk_document(doc, chunker))
    return chunks


def _score(
    documents: list[dict],
    family: str,
    chunker: str,
    model: str,
    m: int,
    construction: int,
    search: int,
    decision: str,
) -> dict:
    run_id = _run_id(chunker, model, m, construction, search)
    existing = {row["run_id"]: row for row in load_rows()}
    if run_id in existing:
        row = dict(existing[run_id])
        print(f"Reusing {run_id} Recall@5={float(row['recall_at_5']):.4f}")
        return row

    n_chunks = _build(documents, run_id, chunker, model, m, construction, search)
    metrics = evaluate(run_id, model)
    row = {
        "run_id": run_id,
        "family": family,
        "chunker": chunker,
        "embedding_model": model,
        "hnsw_m": m,
        "hnsw_construction_ef": construction,
        "hnsw_search_ef": search,
        "n_chunks": n_chunks,
        "decision": decision,
        **metrics,
    }
    append_row(row)
    return row


def _build(documents, run_id, chunker, model, m, construction, search) -> int:
    chunks = _chunks_for(documents, chunker)
    print(f"Indexing {run_id} ({len(chunks)} chunks)")
    vectors = embed_chunks(chunks, model)
    build_index(run_id, chunks, vectors, m, construction, search)
    return len(chunks)


def _drop_index(run_id: str) -> None:
    path = index_dir(run_id)
    if not path.exists():
        return
    for attempt in range(6):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    print(f"Warning: could not delete index {path}; continuing")


def _compete(documents, family: str, incumbent: dict, challengers: list[tuple]) -> dict:
    current = incumbent
    for chunker, model, m, construction, search in challengers:
        row = _score(documents, family, chunker, model, m, construction, search, "pending")
        if beats(row, current):
            print(
                f"Keep {row['run_id']}: Recall@5 {float(row['recall_at_5']):.4f} "
                f"> {float(current['recall_at_5']):.4f} beyond noise"
            )
            if row["run_id"] != current["run_id"]:
                _drop_index(current["run_id"])
                update_decision(current["run_id"], "replaced")
            row["decision"] = "kept"
            update_decision(row["run_id"], "kept")
            current = row
        else:
            gain = float(row["recall_at_5"]) - float(current["recall_at_5"])
            label = "within_noise" if gain >= 0 else "worse"
            print(f"Discard {row['run_id']} ({label}, gain={gain:.4f})")
            row["decision"] = label
            update_decision(row["run_id"], label)
            if row["run_id"] != current["run_id"]:
                _drop_index(row["run_id"])
    return current


def _seed_incumbent(documents: list[dict]) -> dict:
    """Resume from the best scored row when experiments.csv already has results."""
    prior = load_rows()
    if prior:
        best = max(prior, key=lambda row: float(row["recall_at_5"]))
        print(
            f"Seeding from best prior run {best['run_id']} "
            f"Recall@5={float(best['recall_at_5']):.4f}"
        )
        return best
    return _score(
        documents,
        "chunking",
        INCUMBENT_CHUNKER,
        INCUMBENT_MODEL,
        DEFAULT_M,
        DEFAULT_CONSTRUCTION,
        DEFAULT_SEARCH,
        "incumbent",
    )


def ablate() -> dict:
    documents = load_corpus()
    models = _available_models()
    print("Embedding models:", ", ".join(models))
    incumbent = _seed_incumbent(documents)
    chunk_challengers = [
        (name, INCUMBENT_MODEL, DEFAULT_M, DEFAULT_CONSTRUCTION, DEFAULT_SEARCH)
        for name in CHUNKERS
        if name != incumbent.get("chunker")
    ]
    winner = _compete(documents, "chunking", incumbent, chunk_challengers)

    embed_challengers = [
        (winner["chunker"], model, DEFAULT_M, DEFAULT_CONSTRUCTION, DEFAULT_SEARCH)
        for model in models
        if model != winner["embedding_model"]
    ]
    winner = _compete(documents, "embedding", winner, embed_challengers)

    hnsw_challengers = [
        (winner["chunker"], winner["embedding_model"], m, construction, search)
        for m, construction, search in HNSW
        if not (
            int(m) == int(winner["hnsw_m"])
            and int(construction) == int(winner["hnsw_construction_ef"])
            and int(search) == int(winner["hnsw_search_ef"])
        )
    ]
    winner = _compete(documents, "hnsw", winner, hnsw_challengers)
    if not index_dir(winner["run_id"]).exists():
        _build(
            documents,
            winner["run_id"],
            winner["chunker"],
            winner["embedding_model"],
            int(winner["hnsw_m"]),
            int(winner["hnsw_construction_ef"]),
            int(winner["hnsw_search_ef"]),
        )
    payload = {
        "run_id": winner["run_id"],
        "chunker": winner["chunker"],
        "embedding_model": winner["embedding_model"],
        "hnsw_m": int(winner["hnsw_m"]),
        "hnsw_construction_ef": int(winner["hnsw_construction_ef"]),
        "hnsw_search_ef": int(winner["hnsw_search_ef"]),
        "recall_at_5": float(winner["recall_at_5"]),
        "recall_se": float(winner["recall_se"]),
        "noise_floor": float(winner["noise_floor"]),
        "p50_latency_ms": float(winner["p50_latency_ms"]),
        "n_chunks": int(float(winner["n_chunks"])),
        "index_bytes": int(float(winner["index_bytes"])),
        "rule": "Replace the incumbent only when Recall@5 gain exceeds 2 * sqrt(se_a^2 + se_b^2). Latency never overrides Recall@5.",
    }
    WINNER.parent.mkdir(parents=True, exist_ok=True)
    WINNER.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Winner {payload['run_id']} Recall@5={payload['recall_at_5']:.4f}")
    return payload
