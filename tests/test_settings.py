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
        self.assertEqual(settings.INGESTION_WORKER_CONCURRENCY, 2)

    def test_database_pool_defaults_are_valid(self):
        config = Settings()
        config.validate_knowledge_base_settings()
        self.assertEqual(config.DB_POOL_MIN_SIZE, 1)
        self.assertEqual(config.DB_POOL_MAX_SIZE, 3)

    def test_invalid_database_pool_settings_are_rejected(self):
        invalid_configs = (
            {"DB_POOL_MIN_SIZE": -1},
            {"DB_POOL_MAX_SIZE": 0},
            {"DB_POOL_MIN_SIZE": 4, "DB_POOL_MAX_SIZE": 3},
            {"DB_POOL_TIMEOUT_SECONDS": 0},
            {"DB_CONNECT_TIMEOUT_SECONDS": 0},
            {"DB_STATEMENT_TIMEOUT_MS": 0},
        )
        for overrides in invalid_configs:
            with self.subTest(overrides=overrides):
                config = Settings()
                for name, value in overrides.items():
                    setattr(config, name, value)
                with self.assertRaises(ValueError):
                    config.validate_knowledge_base_settings()

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
