"""Verify chunk metadata reaches Chroma without overriding memory identity."""
from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch


try:
    from langchain_core.documents import Document
except ImportError:
    @dataclass
    class Document:
        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)

    core = types.ModuleType("langchain_core")
    documents_module = types.ModuleType("langchain_core.documents")
    documents_module.Document = Document
    core.documents = documents_module
    sys.modules.setdefault("langchain_core", core)
    sys.modules.setdefault("langchain_core.documents", documents_module)

try:
    import langchain_chroma  # noqa: F401
except ImportError:
    chroma_module = types.ModuleType("langchain_chroma")
    chroma_module.Chroma = object
    sys.modules.setdefault("langchain_chroma", chroma_module)

try:
    import rag.embeddings  # noqa: F401
except ImportError:
    embeddings_module = types.ModuleType("rag.embeddings")
    embeddings_module.get_embeddings = lambda: None
    sys.modules["rag.embeddings"] = embeddings_module

from rag.vector_store import index_memory


class _Collection:
    def __init__(self):
        self.kwargs = None

    def add(self, **kwargs):
        self.kwargs = kwargs


class _VectorStore:
    def __init__(self):
        self._collection = _Collection()
        self.deleted = []

    def delete(self, **kwargs):
        self.deleted.append(kwargs)


class _Embeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


class TestPhase4VectorMetadata(unittest.TestCase):
    def test_source_context_is_indexed_and_identity_fields_are_authoritative(self):
        store = _VectorStore()
        chunks = [Document(
            page_content="Flight AB123 departs at 10:30 AM.",
            metadata={
                "source_page": 2,
                "section_title": "Flight Details",
                "document_type": "pdf",
                "pages": [{"page": 2}],
                "memory_id": "untrusted-memory-id",
                "user_id": "untrusted-user-id",
            },
        )]
        with patch("rag.vector_store.get_vector_store", return_value=store), \
             patch("rag.vector_store.get_embeddings", return_value=_Embeddings()):
            count = index_memory("memory-123", "user-456", "document", chunks)

        self.assertEqual(count, 1)
        metadata = store._collection.kwargs["metadatas"][0]
        self.assertEqual(metadata["source_page"], 2)
        self.assertEqual(metadata["section_title"], "Flight Details")
        self.assertEqual(metadata["document_type"], "pdf")
        self.assertEqual(metadata["memory_id"], "memory-123")
        self.assertEqual(metadata["user_id"], "user-456")
        self.assertEqual(metadata["memory_type"], "document")
        self.assertEqual(metadata["chunk_index"], 0)
        self.assertNotIn("pages", metadata)


if __name__ == "__main__":
    unittest.main()
