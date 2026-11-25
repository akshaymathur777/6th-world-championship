# Local CDC guideline retrieval without sending the patient question off-machine

**Track:** Healthcare & Life Sciences  
**Team:** (fill before Kaggle submit)  
**Deadline buffer submission date:** 28 March 2027 UTC

## Problem

Clinical questions often contain identifiers and clinical detail that should not leave the institution. Public practice guidelines, by contrast, are meant to be shared. The privacy-first design is therefore asymmetric: keep the question and the retrieval stack local; publish only the public corpus and the index built from it.

## Solution

1. Ingest the redistributable CDC slice of [epfl-llm/guidelines](https://huggingface.co/datasets/epfl-llm/guidelines) (616 documents after dropping third-party notices and empty rows).
2. Chunk, embed with a local Ollama model, and index in Chroma (`PersistentClient`, cosine HNSW).
3. Score every change with one metric: **Recall@5** on a frozen 200-question set.
4. Export a YankeeAI Desktop v3 collection zip (`yankeeai-collection-export`) whose `chromadb/` directory and `manifest.json` match the 0.1.0 backend import contract.
5. Demo in Yankee: import the zip, chat with the collection, cite section titles. No cloud LLM, no Chroma Cloud.

```
CDC guidance → section-aware chunks → local Chroma
                                      ↓
                              Recall@5 (frozen eval)
                                      ↓
                         Yankee import ZIP → local answer with citations
Patient question ───────────────────────────→ (never published)
```

## Corpus

| Item | Value |
| --- | --- |
| Source | `epfl-llm/guidelines`, `source=cdc` |
| Kept | 616 documents / 6.77M words |
| Dropped | 5 (courtesy / third-party or too short) |
| Title / URL | Recovered from headings and the first `cdc.gov` link (upstream CDC fields are empty) |
| License stance | U.S. government works only; opening-window third-party notices rejected |

## Evaluation protocol

`chymaabs/CDCQA` was not on the Hugging Face Hub on 2026-11-17, so questions were drafted once from CDC sections (80–400 words), sorted by `sha256(doc_id, section)`, and locked in `eval/FREEZE.json`. The scorer refuses to run if the corpus or question file changes after freeze.

A top-5 hit is either:

- same `url` and `section` as the gold label, or
- the gold evidence quote appears in a retrieved document.

Noise floor: `2 × bootstrap SE` of the hit indicators. Latency never keeps a change that loses Recall@5.

## Ablation results

Replace the incumbent only when the Recall@5 gain clears `2√(se_a² + se_b²)`. Full log: `experiments.csv`.

| Family | Config | Recall@5 | SE | Decision |
| --- | --- | --- | --- | --- |
| chunking | `fixed_256` + `nomic-embed-text` | 0.530 | 0.035 | worse |
| chunking | `fixed_128` + `nomic-embed-text` | 0.550 | 0.034 | worse |
| chunking | `section` + `nomic-embed-text` | 0.645 | 0.034 | replaced later |
| chunking | `recommendation` + `nomic-embed-text` | 0.645 | 0.034 | within noise |
| embedding | `section` + `all-minilm` | 0.655 | 0.034 | within noise |
| embedding | `section` + `mxbai-embed-large` | **0.760** | 0.031 | **kept** |
| hnsw | M=32, construction=128, search=64 | 0.760 | 0.031 | within noise |
| hnsw | M=16, construction=200, search=200 | 0.765 | 0.031 | within noise |

**Winner:** `section` chunking, `mxbai-embed-large`, default HNSW (M=16, ef=100). Recall@5 = 0.760 on 200 frozen questions (p50 latency 33 ms). HNSW retunes did not clear the noise floor, so defaults stay.

## Yankee integration

YankeeAI Desktop 0.1.0 imports:

- **Standard zip:** `manifest.json` with `format=yankeeai-collection-export`, `version=3`, `embedding_model_id=ollama:<tag>`, native `chromadb/`, and `files/<id>/…`.
- **Raw zip:** no manifest; must contain `chroma.sqlite3`; the UI asks for name and embedding model.

Query-time embedding uses the same Ollama `/api/embed` path with `truncate: true`. See [YANKEE_IMPORT.md](YANKEE_IMPORT.md).

## Limitations

- Not a medical device; not clinical advice; no randomized clinical validation.
- Snapshot of CDC text packaged upstream (knowledge cutoff ~August 2023).
- Only ~38% of documents recovered a `cdc.gov` URL; section matching still works via recovered titles and section headings.
- Eval questions are model-drafted from the corpus, not an external clinician gold set.

## Reproducibility

```powershell
python -m src all
```

Artifacts:

- Code: https://github.com/akshaymathur777/6th-world-championship
- Dataset: https://huggingface.co/datasets/akshaymathur777/cdc-clinical-guidelines-privacy-first
- Yankee import: `cdc-guidelines-yankee.zip` with embedding model `ollama:mxbai-embed-large`
- Cover: `docs/cover.png`
- Demo transcript: `docs/demo_transcript.md`
- Video: `docs/championship_video.mp4` (~7.6 min Part 1 + Part 2 slideshow; upload to YouTube)
