"""Local Ollama client. Embeddings and question writing stay on this machine."""

from __future__ import annotations

import os

import httpx

HOST_ENV = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip() or "http://127.0.0.1:11434"
if not HOST_ENV.startswith(("http://", "https://")):
    HOST_ENV = "http://" + HOST_ENV
HOST_ENV = HOST_ENV.replace("://0.0.0.0", "://127.0.0.1")
HOST = HOST_ENV.rstrip("/")


def embed(texts: list[str], model: str, timeout: float = 300) -> list[list[float]]:
    if not texts:
        return []
    response = httpx.post(
        f"{HOST}/api/embed",
        json={"model": model, "input": texts, "truncate": True},
        timeout=timeout,
    )
    response.raise_for_status()
    vectors = response.json().get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise RuntimeError(f"Ollama returned {0 if not vectors else len(vectors)} vectors for {len(texts)} texts")
    return vectors


def chat(model: str, system: str, user: str, num_predict: int = 80) -> str:
    response = httpx.post(
        f"{HOST}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0, "num_predict": num_predict},
        },
        timeout=180,
    )
    response.raise_for_status()
    return (response.json().get("message") or {}).get("content") or ""
