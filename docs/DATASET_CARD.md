# Dataset: local CDC clinical guidelines for YankeeAI

Redistributable CDC clinical practice guidance plus a local Chroma index for privacy-first retrieval.

## Contents

| Path | Description |
| --- | --- |
| `cdc.jsonl` | 616 CDC documents (title, url, text) after third-party drops |
| `dropped.jsonl` | Drop reasons for excluded rows |
| `corpus_manifest.json` | Counts and SHA-256 of the frozen corpus |
| `questions.jsonl` | Frozen 200-question eval set |
| `FREEZE.json` | Locks corpus + question hashes |
| `cdc-guidelines-yankee.zip` | YankeeAI Desktop v3 collection export |
| `cdc-guidelines-raw-chroma.zip` | Raw Chroma archive (no manifest) |
| `winner.json` | Winning chunker / embedding / HNSW config |

## Source

CDC slice of [epfl-llm/guidelines](https://huggingface.co/datasets/epfl-llm/guidelines). U.S. government works; pages with a third-party notice in the opening were dropped.

## Intended use

Local RAG with YankeeAI Desktop or Chroma. Not a medical device. Not clinical advice.
