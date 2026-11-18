# Local CDC guideline retrieval for Privacy-First Agentic AI

A clinician asks about a specific patient. That question never leaves the machine. Retrieval runs over public CDC guidance in a local Chroma database, and YankeeAI Desktop answers with citations back to the source section.

This is the engineering package for the [6th World Championship in Data Science & AI](https://www.kaggle.com/competitions/privacy-first-agentic-ai) (Privacy-First Agentic AI with Yankee AI and Chroma DB).

## Privacy boundary

| Stays local | Published |
| --- | --- |
| Patient / clinician question | Public CDC guideline text |
| Query embedding | Persistent Chroma index |
| Yankee chat transcript | Frozen eval questions (no PHI) |

## One number

**Recall@5** from `python -m src evaluate` (via `python -m src ablate`). A hit is a top-5 chunk that either matches the gold document URL and section, or contains the gold evidence quote. Latency and index size are logged and never override Recall@5. A challenger replaces the incumbent only when the gain clears `2 * sqrt(se_a² + se_b²)`.

## Setup

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
# Ollama must be running with at least nomic-embed-text
ollama pull nomic-embed-text
ollama pull llama3.2:3b   # only needed to freeze the question set once
```

## Rebuild

```powershell
.\.venv\Scripts\python.exe -m src ingest
.\.venv\Scripts\python.exe -m src freeze-eval   # no-op once eval/FREEZE.json exists
.\.venv\Scripts\python.exe -m src ablate
.\.venv\Scripts\python.exe -m src export
.\.venv\Scripts\python.exe -m src demo
```

Or `python -m src all`.

## Yankee import

1. Install [YankeeAI Desktop 0.1.0](https://github.com/isods/yankeeai).
2. Import `data/exports/cdc-guidelines-yankee.zip` (standard v3 export) or the raw Chroma zip.
3. Use embedding model `ollama:mxbai-embed-large` (winner in `data/winner.json`; Recall@5 = 0.760).
4. **Chat with collection**.

Details: [docs/YANKEE_IMPORT.md](docs/YANKEE_IMPORT.md).

## Winner

`section` chunking + `mxbai-embed-large` + default HNSW → **Recall@5 = 0.760** (n=200, SE≈0.031). Published dataset: [akshaymathur777/cdc-clinical-guidelines-privacy-first](https://huggingface.co/datasets/akshaymathur777/cdc-clinical-guidelines-privacy-first).

## Layout

| Path | Role |
| --- | --- |
| `src/` | Ingest, chunk, index, evaluate, ablate, Yankee export |
| `eval/questions.jsonl` | Frozen 200-question set |
| `eval/FREEZE.json` | Locks corpus + question hashes |
| `experiments.csv` | Every indexed run and its Recall@5 |
| `data/winner.json` | Winning config after ablation |
| `data/exports/` | Yankee v3 zip + raw Chroma zip |
| `docs/WRITEUP.md` | Kaggle writeup source |
| `docs/VIDEO_SCRIPT.md` | ≤10 minute YouTube outline |
| `docs/cover.png` | Writeup cover image |

## License note

CDC web content used here is U.S. government work (public domain). Upstream packaging is [epfl-llm/guidelines](https://huggingface.co/datasets/epfl-llm/guidelines). Documents with a third-party notice in the opening are dropped. This is a retrieval demonstration, not a medical device, and not clinical advice.
