"""Build a persistent Chroma collection from precomputed embeddings."""

from __future__ import annotations

import os
import shutil

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
import numpy as np

from src.chunk import Chunk
from src.paths import INDEXES

ADD_BATCH = 256


def index_dir(run_id: str):
    return INDEXES / run_id


def build_index(
    run_id: str,
    chunks: list[Chunk],
    vectors: np.ndarray,
    hnsw_m: int,
    construction_ef: int,
    search_ef: int,
) -> None:
    path = index_dir(run_id)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(path))
    try:
        collection = client.create_collection(
            name="cdc_guidelines",
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": int(hnsw_m),
                "hnsw:construction_ef": int(construction_ef),
                "hnsw:search_ef": int(search_ef),
            },
        )
        for start in range(0, len(chunks), ADD_BATCH):
            batch = chunks[start : start + ADD_BATCH]
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.embedded_document() for chunk in batch],
                metadatas=[chunk.metadata() for chunk in batch],
                embeddings=vectors[start : start + len(batch)].tolist(),
            )
    finally:
        client.close()


def open_collection(run_id: str):
    client = chromadb.PersistentClient(path=str(index_dir(run_id)))
    collection = client.get_collection("cdc_guidelines", embedding_function=None)
    return client, collection


def directory_size(path) -> int:
    total = 0
    if not path.exists():
        return 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total
