"""Run the three frozen demo questions against the winning local index."""

from __future__ import annotations

import json

from src.embed import embed_query
from src.evaluate import load_questions
from src.index import open_collection
from src.ollama import chat
from src.paths import DOCS, WINNER

SYSTEM = (
    "You answer a clinician using only the retrieved CDC guidance excerpts. "
    "Cite the section title for each claim. If the excerpts do not answer the question, say so. "
    "This is a retrieval demonstration, not a medical device, and it is not clinical advice. "
    "Do not invent doses, populations, or recommendations that are absent from the excerpts."
)


def demo() -> None:
    winner = json.loads(WINNER.read_text(encoding="utf-8"))
    questions = [row for row in load_questions() if row.get("demo")]
    client, collection = open_collection(winner["run_id"])
    blocks = [
        "# Local retrieval transcript",
        "",
        "These three questions were chosen when the eval set was frozen, before any index was scored.",
        f"Index: `{winner['run_id']}`. Embedding model: `{winner['embedding_model']}` (Ollama, this machine).",
        "",
    ]
    try:
        for question in questions:
            vector = embed_query(question["question"], winner["embedding_model"])
            result = collection.query(
                query_embeddings=[vector],
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )
            metadatas = (result.get("metadatas") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            excerpts = []
            blocks.append(f"## {question['question_id']}")
            blocks.append("")
            blocks.append(question["question"])
            blocks.append("")
            blocks.append("| Rank | Section | Source | Distance |")
            blocks.append("| --- | --- | --- | --- |")
            for rank, (metadata, document, distance) in enumerate(
                zip(metadatas, documents, distances), start=1
            ):
                metadata = metadata or {}
                blocks.append(
                    f"| {rank} | {metadata.get('section', '')} | {metadata.get('url', '')} | {float(distance):.4f} |"
                )
                excerpts.append(
                    f"[{rank}] {metadata.get('title', '')} / {metadata.get('section', '')} "
                    f"({metadata.get('url', '')})\n{document[:900]}"
                )
            answer = chat(
                "llama3.2:3b",
                SYSTEM,
                "Question:\n"
                + question["question"]
                + "\n\nExcerpts:\n"
                + "\n\n".join(excerpts),
                num_predict=220,
            ).strip()
            blocks.append("")
            blocks.append(answer)
            blocks.append("")
    finally:
        client.close()

    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "demo_transcript.md"
    path.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
    print(f"Wrote {path}")
