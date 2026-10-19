"""Focused contracts for configurable parallel preparation workers."""
import unittest
from unittest.mock import patch

from memory import background_worker


class ParallelWorkerContractsTests(unittest.TestCase):
    def test_default_worker_concurrency_is_two(self):
        with patch.object(background_worker.settings, "INGESTION_WORKER_CONCURRENCY", 2, create=True):
            self.assertEqual(background_worker._worker_count(), 2)

    def test_worker_concurrency_is_at_least_one(self):
        with patch.object(background_worker.settings, "INGESTION_WORKER_CONCURRENCY", 0, create=True):
            self.assertEqual(background_worker._worker_count(), 1)

    def test_worker_concurrency_is_capped_to_protect_local_resources(self):
        with patch.object(background_worker.settings, "INGESTION_WORKER_CONCURRENCY", 99, create=True):
            self.assertEqual(background_worker._worker_count(), 8)

    def test_invalid_worker_concurrency_uses_safe_default(self):
        with patch.object(background_worker.settings, "INGESTION_WORKER_CONCURRENCY", "invalid", create=True):
            self.assertEqual(background_worker._worker_count(), 2)


if __name__ == "__main__":
    unittest.main()
