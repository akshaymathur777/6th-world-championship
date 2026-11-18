"""Append-only experiment log. One row per indexed run."""

from __future__ import annotations

import csv
from datetime import datetime, timezone

from src.paths import EXPERIMENTS

FIELDS = [
    "run_id",
    "created_utc",
    "family",
    "chunker",
    "embedding_model",
    "hnsw_m",
    "hnsw_construction_ef",
    "hnsw_search_ef",
    "n_chunks",
    "n_questions",
    "recall_at_5",
    "recall_se",
    "noise_floor",
    "section_hit_rate",
    "quote_hit_rate",
    "p50_latency_ms",
    "index_bytes",
    "decision",
]


def load_rows() -> list[dict]:
    if not EXPERIMENTS.exists():
        return []
    with EXPERIMENTS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_row(row: dict) -> None:
    write_header = not EXPERIMENTS.exists() or EXPERIMENTS.stat().st_size == 0
    row = {"created_utc": datetime.now(timezone.utc).isoformat(), **row}
    with EXPERIMENTS.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def noise_gap(left: dict, right: dict) -> float:
    import math

    return 2 * math.sqrt(float(left["recall_se"]) ** 2 + float(right["recall_se"]) ** 2)


def update_decision(run_id: str, decision: str) -> None:
    rows = load_rows()
    for row in rows:
        if row["run_id"] == run_id:
            row["decision"] = decision
    with EXPERIMENTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def beats(challenger: dict, incumbent: dict) -> bool:
    """A challenger replaces the incumbent only when Recall@5 clears the noise floor."""
    gain = float(challenger["recall_at_5"]) - float(incumbent["recall_at_5"])
    return gain > noise_gap(challenger, incumbent)
