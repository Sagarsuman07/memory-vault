"""Phase 5 tests; run in the project's mvenv with Chroma/LangChain installed."""
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
import rag.vector_store as vs


class FakeCollection:
    def __init__(self):
        self.records = {}
        self.fail_next_upsert = False

    def get(self, ids=None, where=None, include=None, limit=None):
        selected = list(self.records)
        if ids is not None:
            wanted = set(ids)
            selected = [key for key in selected if key in wanted]
        if where:
            selected = [key for key in selected if all(
                self.records[key]["metadatas"].get(k) == v for k, v in where.items())]
        if limit is not None:
            selected = selected[:limit]
        return {
            "ids": selected,
            "documents": [self.records[k]["documents"] for k in selected]
                if include is None or "documents" in include else None,
            "metadatas": [self.records[k]["metadatas"] for k in selected]
                if include is None or "metadatas" in include else None,
            "embeddings": [self.records[k]["embeddings"] for k in selected]
                if include is None or "embeddings" in include else None,
        }

    def upsert(self, ids, embeddings, documents, metadatas):
        if self.fail_next_upsert:
            self.fail_next_upsert = False
            raise RuntimeError("simulated Chroma write failure")
        for rid, vector, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.records[rid] = {"embeddings": list(vector), "documents": document,
                                 "metadatas": dict(metadata)}

    def delete(self, ids=None, where=None):
        if ids is not None:
            for rid in ids:
                self.records.pop(rid, None)
        elif where:
            for rid in list(self.records):
                if all(self.records[rid]["metadatas"].get(k) == v for k, v in where.items()):
                    self.records.pop(rid, None)


class FakeStore:
    def __init__(self, collection):
        self._collection = collection


class FakeEmbeddings:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)) + i for i in range(self.dimension)] for t in texts]


class Phase5IndexingTests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection()
        self.embedder = FakeEmbeddings()
        self.store_patch = patch.object(vs, "get_vector_store", return_value=FakeStore(self.collection))
        self.embedding_patch = patch.object(vs, "get_embeddings", return_value=self.embedder)
        self.batch_patch = patch.object(vs.settings, "EMBEDDING_BATCH_SIZE", 2, create=True)
        self.store_patch.start(); self.embedding_patch.start(); self.batch_patch.start()
        self.addCleanup(self.store_patch.stop)
        self.addCleanup(self.embedding_patch.stop)
        self.addCleanup(self.batch_patch.stop)

    def test_embedding_batches_and_deterministic_ids(self):
        chunks = [Document(page_content=f"chunk {i}", metadata={"section_title": "Test"}) for i in range(5)]
        self.assertEqual(vs.index_memory("memory-a", "user-a", "document", chunks), 5)
        self.assertEqual([len(call) for call in self.embedder.calls], [2, 2, 1])
        self.assertEqual(set(self.collection.records), {f"memory-a_chunk_{i}" for i in range(5)})
        self.assertTrue(vs.verify_memory_index("memory-a", chunks)["valid"])

    def test_reindex_is_idempotent_and_removes_obsolete_chunks(self):
        vs.index_memory("memory-a", "user-a", "document", [Document(page_content="old"), Document(page_content="old 2")])
        new = [Document(page_content="updated")]
        vs.index_memory("memory-a", "user-a", "document", new)
        vs.index_memory("memory-a", "user-a", "document", new)
        self.assertEqual(set(self.collection.records), {"memory-a_chunk_0"})
        self.assertEqual(self.collection.records["memory-a_chunk_0"]["documents"], "updated")

    def test_failed_replacement_restores_previous_records(self):
        vs.index_memory("memory-a", "user-a", "document", [Document(page_content="old 0"), Document(page_content="old 1")])
        before = {k: {**v, "embeddings": list(v["embeddings"]), "metadatas": dict(v["metadatas"])} for k, v in self.collection.records.items()}
        self.collection.fail_next_upsert = True
        with self.assertRaisesRegex(RuntimeError, "simulated Chroma write failure"):
            vs.index_memory("memory-a", "user-a", "document", [Document(page_content="new")])
        self.assertEqual(self.collection.records, before)

    def test_dimension_mismatch_is_rejected_before_write(self):
        vs.index_memory("memory-a", "user-a", "document", [Document(page_content="old")])
        before = {k: {**v, "embeddings": list(v["embeddings"]), "metadatas": dict(v["metadatas"])} for k, v in self.collection.records.items()}
        self.embedder.dimension = 2
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            vs.index_memory("memory-b", "user-b", "document", [Document(page_content="new")])
        self.assertEqual(self.collection.records, before)

    def test_non_finite_embedding_is_rejected(self):
        class BadEmbeddings:
            def embed_documents(self, texts):
                return [[float("nan"), 0.0, 1.0] for _ in texts]
        with patch.object(vs, "get_embeddings", return_value=BadEmbeddings()):
            with self.assertRaisesRegex(ValueError, "finite numeric"):
                vs.index_memory("memory-a", "user-a", "document", [Document(page_content="x")])
        self.assertEqual(self.collection.records, {})

    def test_empty_chunks_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "no chunks"):
            vs.index_memory("memory-a", "user-a", "document", [])

    def test_application_identity_metadata_wins(self):
        chunk = Document(page_content="hello", metadata={"memory_id": "wrong", "user_id": "wrong", "chunk_index": 999, "source_page": 3})
        vs.index_memory("memory-a", "user-a", "document", [chunk])
        metadata = self.collection.records["memory-a_chunk_0"]["metadatas"]
        self.assertEqual((metadata["memory_id"], metadata["user_id"], metadata["chunk_index"]), ("memory-a", "user-a", 0))
        self.assertEqual(metadata["source_page"], 3)


if __name__ == "__main__":
    unittest.main()
