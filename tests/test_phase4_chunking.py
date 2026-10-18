"""Phase 4 chunking tests; use small dependency stubs only if LangChain is absent."""
from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Any


try:
    import langchain_core.documents  # noqa: F401
    import langchain_text_splitters  # noqa: F401
except ImportError:
    @dataclass
    class _Document:
        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)

    core = types.ModuleType("langchain_core")
    documents = types.ModuleType("langchain_core.documents")
    documents.Document = _Document
    core.documents = documents
    sys.modules.setdefault("langchain_core", core)
    sys.modules.setdefault("langchain_core.documents", documents)

    splitters = types.ModuleType("langchain_text_splitters")

    class _FallbackRecursiveSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def create_documents(self, texts, metadatas=None):
            results = []
            for source_index, text in enumerate(texts):
                metadata = dict((metadatas or [{}])[source_index])
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    content = text[start:end].strip()
                    if content:
                        results.append(_Document(content, dict(metadata)))
                    if end >= len(text):
                        break
                    start = max(start + 1, end - self.chunk_overlap)
            return results

    splitters.RecursiveCharacterTextSplitter = _FallbackRecursiveSplitter
    sys.modules.setdefault("langchain_text_splitters", splitters)

from config.settings import settings
from rag.chunker import split_text


class TestPhase4Chunking(unittest.TestCase):
    def setUp(self):
        self.original_size = settings.CHUNK_SIZE
        self.original_overlap = settings.CHUNK_OVERLAP

    def tearDown(self):
        settings.CHUNK_SIZE = self.original_size
        settings.CHUNK_OVERLAP = self.original_overlap

    def test_empty_and_whitespace_input_return_no_chunks(self):
        self.assertEqual(split_text(""), [])
        self.assertEqual(split_text(" \n\t "), [])

    def test_chunk_size_and_overlap_are_read_from_settings(self):
        settings.CHUNK_SIZE = 80
        settings.CHUNK_OVERLAP = 12
        text = " ".join(f"token{i}" for i in range(80))
        chunks = split_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk.page_content) <= 80 for chunk in chunks))
        self.assertIn("token0", " ".join(chunk.page_content for chunk in chunks))
        self.assertIn("token79", " ".join(chunk.page_content for chunk in chunks))

    def test_invalid_runtime_chunk_configuration_is_rejected(self):
        settings.CHUNK_SIZE = 10
        settings.CHUNK_OVERLAP = 10
        with self.assertRaisesRegex(ValueError, "CHUNK_OVERLAP"):
            split_text("some useful text")

    def test_pdf_page_context_and_scalar_metadata_are_preserved(self):
        settings.CHUNK_SIZE = 100
        settings.CHUNK_OVERLAP = 10
        text = "[Page 1]\nFlight AB123 departs at 10:30 AM.\n\n[Page 2]\nHotel price is ₹8,500."
        chunks = split_text(text, metadata={
            "document_type": "pdf",
            "extraction_method": "pymupdf_text",
            "title": "Travel Notes",
            "pages": [{"page": 1}],
        })
        page_one = [chunk for chunk in chunks if chunk.metadata.get("source_page") == 1]
        page_two = [chunk for chunk in chunks if chunk.metadata.get("source_page") == 2]
        self.assertTrue(page_one)
        self.assertTrue(page_two)
        self.assertTrue(all(chunk.metadata["document_type"] == "pdf" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["title"] == "Travel Notes" for chunk in chunks))
        self.assertTrue(all("pages" not in chunk.metadata for chunk in chunks))
        self.assertIn("AB123", " ".join(chunk.page_content for chunk in page_one))
        self.assertIn("8,500", " ".join(chunk.page_content for chunk in page_two))

    def test_markdown_heading_is_attached_as_section_context(self):
        chunks = split_text("# Trip Details\nFlight AB123\n\n## Hotel\nHotel price: 8500")
        hotel_chunks = [chunk for chunk in chunks if chunk.metadata.get("section_title") == "Hotel"]
        trip_chunks = [chunk for chunk in chunks if chunk.metadata.get("section_title") == "Trip Details"]
        self.assertTrue(hotel_chunks)
        self.assertTrue(trip_chunks)
        self.assertIn("8500", " ".join(chunk.page_content for chunk in hotel_chunks))

    def test_exact_duplicate_chunks_are_flagged_not_silently_removed(self):
        text = "[Page 1]\nRepeated header\n\n[Page 2]\nRepeated header"
        chunks = split_text(text)
        repeated = [chunk for chunk in chunks if chunk.page_content == "Repeated header"]
        self.assertEqual(len(repeated), 2)
        self.assertNotIn("duplicate_content", repeated[0].metadata)
        self.assertTrue(repeated[1].metadata["duplicate_content"])
        self.assertEqual(repeated[1].metadata["duplicate_occurrence"], 2)

    def test_all_chunks_are_nonempty_and_within_configured_size(self):
        settings.CHUNK_SIZE = 64
        settings.CHUNK_OVERLAP = 8
        text = "Paragraph one has important details.\n\n" + ("A long line with identifiers AB-123 and amount ₹500. " * 12)
        chunks = split_text(text)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.page_content.strip() for chunk in chunks))
        self.assertTrue(all(len(chunk.page_content) <= 64 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
