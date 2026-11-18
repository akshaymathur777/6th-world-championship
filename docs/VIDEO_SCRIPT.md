# YouTube script (≤10 minutes)

Post publicly. No login wall. Structure required by the championship: Part 1 vector DB, Part 2 Yankee demo.

## Part 1 — Vector database (~5 min)

1. **Opening (30s).** One sentence: a clinician question stays on the machine; only public CDC guidance is indexed.
2. **Privacy diagram (45s).** Screen: the mermaid-style flow from the writeup. Point at “question stays local.”
3. **Corpus (60s).** Show `eval/corpus_manifest.json`: 616 CDC docs, 5 dropped with reasons. Open one markdown page and the recovered title/URL.
4. **Chunking (60s).** Show `src/chunk.py` families and a before/after: fixed windows vs section-aware. Mention metadata on every chunk: `url`, `title`, `section`, `chunk_id`.
5. **Metric (60s).** Show `experiments.csv` and the winner Recall@5. Say the noise rule out loud: only keep a change that beats `2√(se²+se²)`.
6. **Embeddings (30s).** Ollama models that Yankee can load; winner id from `data/winner.json`.

## Part 2 — Yankee AI integration (~4–5 min)

1. **Import (60s).** Yankee → My Files → Collections → Import → `cdc-guidelines-yankee.zip`. Select `ollama:nomic-embed-text` (or the winner).
2. **Demo Q1–Q3 (3 min).** Use the three `demo: true` questions from `eval/questions.jsonl` / `docs/demo_transcript.md`. For each: ask, show retrieved section titles, show citations. No patient identifiers on screen.
3. **Close (30s).** Repeat the privacy boundary. Point judges to the public repo, dataset, and writeup.

## Recording checklist

- [ ] Cover image attached on Kaggle writeup
- [ ] Video ≤10:00
- [ ] YouTube public / unlisted without login
- [ ] Part 1 and Part 2 clearly labeled on screen
- [ ] No PHI, no real patient narratives
