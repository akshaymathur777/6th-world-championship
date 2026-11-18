import unittest

from src.chunk import chunk_document, chunk_fixed, chunk_recommendation
from src.evaluate import is_hit
from src.experiments import beats
from src.export_yankee import EXPORT_FORMAT, write_raw_zip, write_v3_zip
from src.textutil import document_title, normalize_text, split_sections, third_party_notice


class TextTests(unittest.TestCase):
    def test_glued_heading_becomes_its_own_section(self):
        raw = "Intro sentence about exposure. More words here so this preamble stays.# BACKGROUND\nAcrylamide details follow in this section with enough words to remain."
        sections = split_sections(raw, min_words=5)
        titles = [section.title for section in sections]
        self.assertIn("BACKGROUND", titles)
        self.assertIn("\n# BACKGROUND", "\n" + normalize_text(raw))

    def test_third_party_notice(self):
        self.assertEqual(
            third_party_notice("This report is being published as a courtesy to the association."),
            "published as a courtesy to a non-CDC organization",
        )
        self.assertIsNone(
            third_party_notice("Copyright 2020 CDC. This guidance is a U.S. government work."),
        )

    def test_title_from_guideline_sentence(self):
        text = (
            "# Summary\n"
            "Preexposure Prophylaxis for HIV Prevention in the United States - 2014: "
            "A Clinical Practice Guideline provides comprehensive information for clinicians today."
        )
        self.assertIn("Clinical Practice Guideline", document_title(text))


class ChunkTests(unittest.TestCase):
    def test_fixed_windows_overlap_and_cover_the_tail(self):
        words = " ".join(f"w{i}" for i in range(100))
        doc = {"doc_id": "d", "title": "T", "url": "cdc:d", "text": words}
        chunks = chunk_fixed(doc, size=30, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertIn("w0", chunks[0].text)
        self.assertIn("w99", chunks[-1].text)

    def test_recommendation_split(self):
        text = (
            "# Treatment\n"
            + " ".join(["intro"] * 50)
            + "\n\nRecommended regimen for adults is listed here with enough extra words.\n\n"
            + " ".join(["follow"] * 40)
        )
        doc = {"doc_id": "d", "title": "T", "url": "cdc:d", "text": text}
        chunks = chunk_recommendation(doc, size=200, overlap=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any(chunk.text.startswith("Recommended") for chunk in chunks))


class MetricTests(unittest.TestCase):
    def test_hit_is_section_or_quote(self):
        question = {
            "url": "https://www.cdc.gov/example",
            "section": "Diagnosis",
            "evidence_quote": "Throat culture is the gold standard diagnostic test.",
        }
        section_hit, quote_hit = is_hit(
            "unrelated",
            {"url": question["url"], "section": "Diagnosis"},
            question,
        )
        self.assertTrue(section_hit)
        self.assertFalse(quote_hit)
        section_hit, quote_hit = is_hit(
            "Notes. Throat culture is the gold standard diagnostic test. More.",
            {"url": "other", "section": "Other"},
            question,
        )
        self.assertFalse(section_hit)
        self.assertTrue(quote_hit)

    def test_noise_rule_rejects_a_small_gain(self):
        incumbent = {"recall_at_5": 0.50, "recall_se": 0.04}
        challenger = {"recall_at_5": 0.53, "recall_se": 0.04}
        self.assertFalse(beats(challenger, incumbent))
        clear = {"recall_at_5": 0.70, "recall_se": 0.03}
        self.assertTrue(beats(clear, incumbent))


class ExportTests(unittest.TestCase):
    def test_v3_zip_matches_yankee_import_contract(self):
        import json
        import os
        import shutil
        import zipfile
        from pathlib import Path

        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        import chromadb

        root = Path("data/tmp/export-test")
        if root.exists():
            shutil.rmtree(root)
        chroma_dir = root / "chroma"
        chroma_dir.mkdir(parents=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.create_collection(
            name="cdc_guidelines",
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "hnsw:M": 16},
        )
        collection.add(
            ids=["d:0:0"],
            documents=["Title: T\nSection: Diagnosis\nThroat culture is the gold standard."],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[
                {
                    "chunk_id": "d:0:0",
                    "doc_id": "d",
                    "file_reference_id": "d",
                    "file_name": "d.md",
                    "title": "T",
                    "url": "cdc:d",
                    "section": "Diagnosis",
                }
            ],
        )
        client.close()
        doc = {"doc_id": "d", "title": "T", "url": "cdc:d", "text": "Throat culture is the gold standard."}
        destination = root / "sample.zip"
        write_v3_zip(chroma_dir, [doc], "nomic-embed-text", destination, chunk_count=1)
        with zipfile.ZipFile(destination) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            names = set(archive.namelist())
        self.assertEqual(manifest["format"], EXPORT_FORMAT)
        self.assertEqual(manifest["version"], 3)
        self.assertEqual(manifest["embedding_model_id"], "ollama:nomic-embed-text")
        self.assertEqual(manifest["vector_export"]["chromadb_dir"], "chromadb")
        self.assertEqual(manifest["files"][0]["file_reference_id"], "d")
        self.assertIn("files/d/d.md", names)
        self.assertTrue(any(name.endswith("chroma.sqlite3") and name.startswith("chromadb/") for name in names))
        raw = root / "raw.zip"
        write_raw_zip(chroma_dir, raw)
        with zipfile.ZipFile(raw) as archive:
            raw_names = archive.namelist()
        self.assertNotIn("manifest.json", raw_names)
        self.assertTrue(any(name.endswith("chroma.sqlite3") for name in raw_names))
        reopened = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            self.assertEqual(reopened.get_collection("cdc_guidelines", embedding_function=None).count(), 1)
        finally:
            reopened.close()
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
