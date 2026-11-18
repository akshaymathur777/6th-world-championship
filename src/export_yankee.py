"""YankeeAI Desktop 0.1.0 collection ZIP.

The layout is taken from `CollectionService._build_collection_export_zip` in the
0.1.0 Windows backend (`yankeeai-backend.exe`). A standard import reads
`manifest.json`. A zip with no manifest is the raw Chroma path and is accepted
only when `chroma.sqlite3` is present.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

COLLECTION_NAME = "cdc-clinical-guidelines"
DESCRIPTION = (
    "Local CDC clinical guidance. Patient questions are embedded on this machine. "
    "This index contains public guidance only."
)
EXPORT_FORMAT = "yankeeai-collection-export"
EXPORT_VERSION = 3


def yankee_model_id(ollama_model: str) -> str:
    """Yankee stores Ollama embedders as ``ollama:<tag>`` and strips that prefix before /api/embed."""
    if ollama_model.startswith("ollama:"):
        return ollama_model
    return f"ollama:{ollama_model}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_entry(doc: dict, uploaded_at: str) -> tuple[str, bytes, dict]:
    body = f"# {doc['title']}\n\nSource: {doc['url']}\n\n{doc['text']}\n".encode("utf-8")
    file_name = f"{doc['doc_id']}.md"
    zip_path = f"files/{doc['doc_id']}/{file_name}"
    entry = {
        "file_reference_id": doc["doc_id"],
        "file_name": file_name,
        "file_type": "text/markdown",
        "file_size": len(body),
        "content_hash": _sha256(body),
        "processing_status": "completed",
        "uploaded_at": uploaded_at,
        "zip_path": zip_path,
    }
    return zip_path, body, entry


def manifest_for(
    documents: list[dict],
    embedding_model: str,
    chunk_count: int,
    uploaded_at: str | None = None,
    run_id: str | None = None,
    recall_at_5: float | None = None,
) -> dict:
    uploaded_at = uploaded_at or datetime.now(timezone.utc).isoformat()
    files = [_file_entry(doc, uploaded_at)[2] for doc in documents]
    manifest = {
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "collection_id": COLLECTION_NAME,
        "name": COLLECTION_NAME,
        "embedding_model_id": yankee_model_id(embedding_model),
        "description": DESCRIPTION,
        "exported_at": uploaded_at,
        "files": files,
        "vector_export": {
            "format": "chromadb_native",
            "chromadb_dir": "chromadb",
            "chunk_count": chunk_count,
        },
    }
    if run_id is not None:
        manifest["run_id"] = run_id
    if recall_at_5 is not None:
        manifest["recall_at_5"] = recall_at_5
    return manifest


def write_v3_zip(
    chroma_dir: Path,
    documents: list[dict],
    embedding_model: str,
    destination: Path,
    chunk_count: int,
    run_id: str | None = None,
    recall_at_5: float | None = None,
) -> None:
    uploaded_at = datetime.now(timezone.utc).isoformat()
    manifest = manifest_for(
        documents,
        embedding_model,
        chunk_count,
        uploaded_at=uploaded_at,
        run_id=run_id,
        recall_at_5=recall_at_5,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        for doc in documents:
            zip_path, body, _ = _file_entry(doc, uploaded_at)
            archive.writestr(zip_path, body)
        for file in chroma_dir.rglob("*"):
            if file.is_file():
                arcname = "chromadb/" + file.relative_to(chroma_dir).as_posix()
                archive.write(file, arcname)


def write_raw_zip(chroma_dir: Path, destination: Path) -> None:
    """No manifest.json. Yankee treats this as a raw Chroma archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in chroma_dir.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(chroma_dir).as_posix())


def export_winner() -> dict:
    from src.index import index_dir
    from src.ingest import load_corpus
    from src.paths import EXPORTS, WINNER

    if not WINNER.exists():
        raise FileNotFoundError("No winner yet. Run: python -m src ablate")
    winner = json.loads(WINNER.read_text(encoding="utf-8"))
    source = index_dir(winner["run_id"])
    if not source.exists():
        raise FileNotFoundError(f"Missing index directory {source}")

    EXPORTS.mkdir(parents=True, exist_ok=True)
    raw_zip = EXPORTS / "cdc-guidelines-raw-chroma.zip"
    yankee_zip = EXPORTS / "cdc-guidelines-yankee.zip"
    if raw_zip.exists():
        raw_zip.unlink()
    if yankee_zip.exists():
        yankee_zip.unlink()
    documents = load_corpus()
    write_raw_zip(source, raw_zip)
    write_v3_zip(
        source,
        documents,
        winner["embedding_model"],
        yankee_zip,
        chunk_count=int(winner["n_chunks"]),
        run_id=winner["run_id"],
        recall_at_5=float(winner["recall_at_5"]),
    )
    result = {
        "raw_chroma_zip": str(raw_zip),
        "yankee_zip": str(yankee_zip),
        "embedding_model_id": yankee_model_id(winner["embedding_model"]),
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "raw_bytes": raw_zip.stat().st_size,
        "yankee_bytes": yankee_zip.stat().st_size,
    }
    (EXPORTS / "export_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Wrote {yankee_zip.name} ({result['yankee_bytes']} bytes) and "
        f"{raw_zip.name} ({result['raw_bytes']} bytes). "
        f"In Yankee, import {yankee_zip.name} with embedding model {result['embedding_model_id']}."
    )
    return result
