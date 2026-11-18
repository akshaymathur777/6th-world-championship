"""Embed chunks with Ollama and cache vectors on disk."""

from __future__ import annotations

import gc
import hashlib
import json
import time

import numpy as np
from tqdm import tqdm

from src.chunk import Chunk
from src.ollama import embed
from src.paths import CACHE

BATCH = 32
CHECKPOINT_EVERY = 20


def _cache_paths(model: str, text_hash: str):
    slug = model.replace(":", "_").replace("/", "_")
    directory = CACHE / "embeddings" / slug
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((model + "\n" + text_hash).encode()).hexdigest()[:16]
    return (
        directory / f"{key}.npy",
        directory / f"{key}.ids.json",
        directory / f"{key}.partial.npy",
        directory / f"{key}.partial.json",
    )


def _content_hash(model: str, chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    for chunk in chunks:
        digest.update(b"\0")
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.embedded_document().encode("utf-8"))
    return digest.hexdigest()


def embed_chunks(chunks: list[Chunk], model: str) -> np.ndarray:
    ids = [chunk.chunk_id for chunk in chunks]
    text_hash = _content_hash(model, chunks)
    array_path, id_path, partial_path, progress_path = _cache_paths(model, text_hash)
    if array_path.exists() and id_path.exists():
        cached_ids = json.loads(id_path.read_text(encoding="utf-8"))
        if cached_ids == ids:
            return np.load(array_path)

    documents = [chunk.embedded_document() for chunk in chunks]
    done = 0
    if partial_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("ids_hash") == hashlib.sha256("\n".join(ids).encode()).hexdigest():
            done = int(progress.get("done", 0))
            print(f"Resuming embeddings at {done}/{len(documents)}")
        else:
            partial_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
            done = 0

    if done == 0:
        probe = _embed_resilient(documents[:1], model)
        dim = len(probe[0])
        memmap = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=(len(documents), dim)
        )
        memmap[0] = probe[0]
        done = 1
        _write_progress(progress_path, ids, done)
    else:
        memmap = np.load(partial_path, mmap_mode="r+")

    starts = list(range(done, len(documents), BATCH))
    for start in tqdm(starts, desc=f"embed {model}", unit="batch"):
        end = min(start + BATCH, len(documents))
        batch = documents[start:end]
        vectors = _embed_resilient(batch, model)
        memmap[start:end] = np.asarray(vectors, dtype=np.float32)
        done = end
        if (start // BATCH) % CHECKPOINT_EVERY == 0 or end == len(documents):
            memmap.flush()
            _write_progress(progress_path, ids, done)

    memmap.flush()
    array = np.array(memmap, dtype=np.float32, copy=True)
    del memmap
    gc.collect()
    np.save(array_path, array)
    id_path.write_text(json.dumps(ids), encoding="utf-8")
    _unlink_retry(partial_path)
    _unlink_retry(progress_path)
    return array


def _unlink_retry(path, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.25 * (attempt + 1))
    path.unlink(missing_ok=True)


def _write_progress(path, ids: list[str], done: int) -> None:
    path.write_text(
        json.dumps(
            {
                "done": done,
                "ids_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def embed_query(text: str, model: str) -> list[float]:
    return embed([text], model)[0]


def _embed_resilient(batch: list[str], model: str) -> list[list[float]]:
    try:
        return embed(batch, model)
    except Exception:
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        return _embed_resilient(batch[:mid], model) + _embed_resilient(batch[mid:], model)
