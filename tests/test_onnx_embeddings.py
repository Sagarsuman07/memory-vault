import unittest
from unittest.mock import patch

import rag.embeddings as embeddings


class OnnxEmbeddingInitializationTests(unittest.TestCase):
    def setUp(self):
        self.previous_embeddings = embeddings._embeddings
        embeddings._embeddings = None
        self.addCleanup(self._restore_embeddings)

    def _restore_embeddings(self):
        embeddings._embeddings = self.previous_embeddings

    @patch("rag.embeddings.HuggingFaceEmbeddings")
    def test_initializes_cpu_onnx_fp32_model_with_batch_size_16(self, embedding_class):
        model = embedding_class.return_value

        result = embeddings.get_embeddings()

        self.assertIs(result, model)
        embedding_class.assert_called_once_with(
            model_name=embeddings.MODEL_NAME,
            model_kwargs={
                "backend": "onnx",
                "device": "cpu",
                "provider": "CPUExecutionProvider",
            },
            encode_kwargs={
                "batch_size": 16,
                "precision": "float32",
                "normalize_embeddings": False,
            },
            query_encode_kwargs={
                "batch_size": 16,
                "precision": "float32",
                "normalize_embeddings": False,
            },
        )

    @patch("rag.embeddings.HuggingFaceEmbeddings")
    def test_reuses_cached_embedding_instance(self, embedding_class):
        first = embeddings.get_embeddings()
        second = embeddings.get_embeddings()

        self.assertIs(first, second)
        embedding_class.assert_called_once()

    @patch("rag.embeddings.HuggingFaceEmbeddings")
    def test_initialization_failure_is_propagated_without_caching(self, embedding_class):
        embedding_class.side_effect = RuntimeError("ONNX initialization failed")

        with self.assertRaisesRegex(RuntimeError, "ONNX initialization failed"):
            embeddings.get_embeddings()

        self.assertIsNone(embeddings._embeddings)


if __name__ == "__main__":
    unittest.main()
