"""Qdrant indexing + hybrid retrieval tests (uses Qdrant's in-memory engine)."""
import hashlib
import math
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from qdrant_client import QdrantClient

import rag.retriever as retriever
import rag.vector_store as vs
from rag.grounding import get_grounded_results

DIM = 384


class FakeEmbeddings:
    """Deterministic bag-of-words embeddings (similar words -> similar vectors)."""

    def _vec(self, text):
        v = [0.0] * DIM
        for token in vs.query_terms(text):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def docs(*texts, **meta):
    return [Document(page_content=t, metadata=dict(meta)) for t in texts]


class QdrantTestCase(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        vs._ensure_collection(self.client)
        self.patches = [
            patch.object(vs, "_CLIENT", self.client),
            patch.object(vs, "get_embeddings", return_value=FakeEmbeddings()),
            patch.object(retriever, "get_embeddings", return_value=FakeEmbeddings()),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.client.close()


class IndexingTests(QdrantTestCase):
    def test_collection_ensures_user_and_memory_payload_indexes(self):
        info = self.client.get_collection(vs.COLLECTION_NAME)
        self.assertIn("user_id", info.payload_schema)
        self.assertIn("memory_id", info.payload_schema)

    def test_existing_collection_missing_payload_index_is_repaired(self):
        self.client.delete_payload_index(vs.COLLECTION_NAME, "user_id", wait=True)
        self.assertNotIn("user_id", self.client.get_collection(vs.COLLECTION_NAME).payload_schema)
        vs._ensure_collection(self.client)
        self.assertIn("user_id", self.client.get_collection(vs.COLLECTION_NAME).payload_schema)

    def test_indexing_embeds_in_bounded_batches(self):
        calls = []

        class TrackingEmbeddings(FakeEmbeddings):
            def embed_documents(self, texts):
                calls.append(len(texts))
                return super().embed_documents(texts)

        with patch.object(vs, "get_embeddings", return_value=TrackingEmbeddings()), \
             patch.object(vs.settings, "EMBEDDING_BATCH_SIZE", 2):
            vs.index_memory(
                "batched", "u1", "document",
                docs("first text", "second text", "third text", "fourth text", "fifth text"),
            )

        self.assertEqual(calls, [2, 2, 1])
        self.assertTrue(vs.verify_memory_index("batched")["valid"])

    def test_index_verify_and_metadata(self):
        chunks = docs("alpha one", "beta two", "gamma three",
                      source_page=3, memory_id="WRONG", user_id="WRONG")
        self.assertEqual(vs.index_memory("m1", "u1", "document", chunks), 3)
        report = vs.verify_memory_index("m1", chunks)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["record_count"], 3)
        self.assertEqual(report["embedding_dimension"], DIM)
        payload = self.client.scroll(vs.COLLECTION_NAME, with_payload=True)[0][0].payload
        self.assertEqual(payload["memory_id"], "m1")      # identity not overridable
        self.assertEqual(payload["user_id"], "u1")
        self.assertEqual(payload["source_page"], 3)

    def test_reindex_replaces_and_removes_stale(self):
        vs.index_memory("m1", "u1", "document", docs("old a", "old b", "old c"))
        new = docs("new a", "new b")
        vs.index_memory("m1", "u1", "document", new)
        vs.index_memory("m1", "u1", "document", new)      # idempotent
        report = vs.verify_memory_index("m1", new)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["record_count"], 2)

    def test_failed_write_removes_new_points(self):
        real_upsert = self.client.upsert

        def boom(*a, **k):
            real_upsert(*a, **k)
            raise RuntimeError("simulated write failure")

        with patch.object(self.client, "upsert", side_effect=boom):
            with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                vs.index_memory("m1", "u1", "document", docs("x one"))
        self.assertEqual(vs.verify_memory_index("m1")["record_count"], 0)

    def test_delete_only_target_memory(self):
        vs.index_memory("m1", "u1", "document", docs("one"))
        vs.index_memory("m2", "u1", "document", docs("two"))
        vs.delete_memory_index("m1")
        self.assertEqual(vs.verify_memory_index("m1")["record_count"], 0)
        self.assertEqual(vs.verify_memory_index("m2")["record_count"], 1)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            vs.index_memory("m1", "u1", "document", [])
        with self.assertRaises(ValueError):
            vs.index_memory("", "u1", "document", docs("x"))
        with self.assertRaises(ValueError):
            vs.index_memory("m1", "u1", "document", docs("   "))

    def test_clone_reuses_vectors_without_embedding(self):
        chunks = docs("hotel booking delhi", "flight ticket mumbai")
        vs.index_memory("src", "u1", "document", chunks)
        with patch.object(vs, "get_embeddings", side_effect=AssertionError("must not embed")):
            n = vs.clone_memory_index(source_memory_id="src", target_memory_id="dup", user_id="u1",
                                      memory_type="document",
                                      expected_texts=[c.page_content for c in chunks])
        self.assertEqual(n, 2)
        self.assertTrue(vs.verify_memory_index("dup", chunks)["valid"])
        self.assertEqual(vs.verify_memory_index("src")["record_count"], 2)

    def test_clone_refuses_changed_text_or_other_user(self):
        vs.index_memory("src", "u1", "document", docs("hotel booking"))
        with self.assertRaises(vs.VectorReuseError):
            vs.clone_memory_index(source_memory_id="src", target_memory_id="dup", user_id="u1",
                                  memory_type="document", expected_texts=["different text"])
        with self.assertRaises(vs.VectorReuseError):
            vs.clone_memory_index(source_memory_id="src", target_memory_id="dup", user_id="u2",
                                  memory_type="document", expected_texts=["hotel booking"])


class RetrievalTests(QdrantTestCase):
    def setUp(self):
        super().setUp()
        vs.index_memory("m1", "u1", "document", docs(
            "Hotel booking confirmation. Hotel ABC in Delhi. Reference ZX9981Q.",
            "Grocery list: milk, eggs, bread and butter.",
            "The electricity bill for March was 1450 rupees.",
        ))
        vs.index_memory("m2", "u1", "document", docs("Passport number P1234567 expires in 2031."))
        vs.index_memory("m3", "u2", "document", docs("Other user's secret hotel booking ZX9981Q."))

    def test_hybrid_finds_exact_identifier(self):
        results = retriever.retrieve("ZX9981Q", "u1", top_k=3)
        self.assertIn("Delhi", results[0][0].page_content)
        self.assertTrue(results[0][0].metadata["keyword_match"])

    def test_hybrid_semantic_overlap(self):
        results = retriever.retrieve("hotel booking delhi", "u1", top_k=3)
        self.assertEqual(results[0][0].metadata["memory_id"], "m1")
        self.assertLess(results[0][1], 1.2)

    def test_user_isolation(self):
        for mode in ("hybrid", "dense", "bm25"):
            results = retriever.retrieve("hotel booking ZX9981Q", "u1", top_k=5, mode=mode)
            self.assertTrue(results)
            self.assertTrue(all(d.metadata["user_id"] == "u1" for d, _ in results), mode)

    def test_memory_scope_isolation(self):
        results = retriever.retrieve("hotel booking passport", "u1", top_k=5, memory_id="m2")
        self.assertTrue(results)
        self.assertTrue(all(d.metadata["memory_id"] == "m2" for d, _ in results))
        self.assertEqual(retriever.retrieve("hotel", "u1", memory_id="m3"), [])  # not u1's

    def test_modes_and_stopword_only_query(self):
        self.assertTrue(retriever.retrieve("passport", "u1", mode="bm25"))
        self.assertTrue(retriever.retrieve("what is the", "u1", mode="dense") is not None)
        self.assertEqual(retriever.retrieve("what is the", "u1", mode="bm25"), [])
        self.assertIsNotNone(retriever.retrieve("what is the", "u1", mode="hybrid"))
        with self.assertRaises(ValueError):
            retriever.retrieve("x", "u1", mode="nope")

    def test_grounding_keeps_keyword_hits_and_drops_noise(self):
        far = Document(page_content="a", metadata={"keyword_match": True})
        noise = Document(page_content="b", metadata={"keyword_match": False})
        near = Document(page_content="c", metadata={"keyword_match": False})
        kept = get_grounded_results([(far, 1.9), (noise, 1.9), (near, 0.5)])
        self.assertEqual([d.page_content for d, _ in kept], ["a", "c"])

    def test_unrelated_question_is_not_grounded(self):
        results = retriever.retrieve("quantum chromodynamics lagrangian", "u1", top_k=3)
        self.assertEqual(get_grounded_results(results), [])

    def test_validation(self):
        self.assertEqual(retriever.retrieve("   ", "u1"), [])
        with self.assertRaises(ValueError):
            retriever.retrieve("x", " ")
        with self.assertRaises(ValueError):
            retriever.retrieve("x", "u1", top_k=0)
        with self.assertRaises(ValueError):
            retriever.retrieve("x", "u1", memory_id=" ")


if __name__ == "__main__":
    unittest.main()
