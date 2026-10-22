import unittest

from config.settings import Settings, settings


class KnowledgeBaseSettingsTests(unittest.TestCase):
    def test_existing_pipeline_defaults_are_preserved(self):
        self.assertEqual(settings.CHUNK_SIZE, 1000)
        self.assertEqual(settings.CHUNK_OVERLAP, 200)
        self.assertEqual(
            settings.EMBEDDING_MODEL,
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertEqual(settings.EMBEDDING_BATCH_SIZE, 32)
        self.assertEqual(settings.INGESTION_MAX_ATTEMPTS, 3)
        self.assertEqual(settings.INGESTION_WORKER_CONCURRENCY, 1)

    def test_invalid_chunk_overlap_is_rejected(self):
        config = Settings()
        config.CHUNK_SIZE = 100
        config.CHUNK_OVERLAP = 100
        with self.assertRaises(ValueError):
            config.validate_knowledge_base_settings()

    def test_invalid_worker_and_retry_values_are_rejected(self):
        config = Settings()
        config.INGESTION_WORKER_CONCURRENCY = 0
        with self.assertRaises(ValueError):
            config.validate_knowledge_base_settings()


if __name__ == "__main__":
    unittest.main()
