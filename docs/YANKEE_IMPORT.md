# YankeeAI import

The collection archive matches YankeeAI Desktop 0.1.0. The layout comes from `CollectionService._build_collection_export_zip` in that build's backend, not from a guessed schema.

## Standard zip

`data/exports/cdc-guidelines-yankee.zip`

- `manifest.json`
  - `format`: `yankeeai-collection-export` (`APP_ID` of `YankeeAI`, plus `-collection-export`)
  - `version`: `3` (2 and 3 are accepted)
  - `embedding_model_id`: `ollama:<tag>`, for example `ollama:nomic-embed-text`
  - `vector_export.format`: `chromadb_native`
  - `vector_export.chromadb_dir`: `chromadb`
  - `files[]`: `file_reference_id`, `file_name`, `file_type`, `file_size`, `content_hash`, `processing_status`, `uploaded_at`, `zip_path`
- `chromadb/chroma.sqlite3` and the other persistent-client files
- `files/<doc_id>/<doc_id>.md` original guidance

Yankee strips the `ollama:` prefix and calls local `POST /api/embed` with `truncate: true`. The index was embedded the same way. In the import dialog, choose that same model if it asks.

## Raw zip

`data/exports/cdc-guidelines-raw-chroma.zip` has no `manifest.json`. Yankee accepts it when `chroma.sqlite3` is in the archive, then asks for a collection name, description, and embedding model. Use the model id printed in `data/winner.json`.

## What was checked

The 0.1.0 MSI was extracted and the backend archive was unpacked. The export and import functions require the fields above. A unit test writes a v3 zip, checks those fields, and reopens the Chroma directory.

Staff embedding models the app already knows how to pull:

- Ollama: `nomic-embed-text`, `mxbai-embed-large`, `bge-m3`
- FastEmbed: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (the app default), `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`

This project only bakes off Ollama models that are installed here, because those are the models Yankee will use to embed the question at query time.

## Winning package

- Zip: `data/exports/cdc-guidelines-yankee.zip`
- Embedding model id: `ollama:mxbai-embed-large`
- Chunker: section-aware (`section`)
- Canonical Recall@5: 0.760 on the frozen 200-question set
