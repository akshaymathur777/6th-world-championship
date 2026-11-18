"""Repository paths. Large derived data stays under data/ and is gitignored."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"
CORPUS = CORPUS_DIR / "cdc.jsonl"
DROPPED = CORPUS_DIR / "dropped.jsonl"
INDEXES = DATA / "indexes"
CACHE = DATA / "cache"
EXPORTS = DATA / "exports"
WINNER = DATA / "winner.json"
EVAL_DIR = ROOT / "eval"
QUESTIONS = EVAL_DIR / "questions.jsonl"
FREEZE = EVAL_DIR / "FREEZE.json"
CORPUS_MANIFEST = EVAL_DIR / "corpus_manifest.json"
EXPERIMENTS = ROOT / "experiments.csv"
DOCS = ROOT / "docs"
