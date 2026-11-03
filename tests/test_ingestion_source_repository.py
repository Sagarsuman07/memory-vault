import unittest
from unittest.mock import Mock, patch

from database.repositories import (
    create_ingestion_source, find_duplicate_content,
    get_ingestion_source, update_ingestion_source,
)


def source_row(**overrides):
    row = {
        "source_id": "source-1", "memory_id": "memory-1", "user_id": "user-a",
        "original_file_name": "notes.txt", "extension": ".txt",
        "media_type": "text/plain", "size_bytes": 4, "sha256": "a" * 64,
        "status": "validated", "stored_file_path": "r2/original/source.txt",
        "duplicate_of_memory_id": None, "warnings_json": "[]",
        "metadata_json": "{}", "error_category": None, "error_message": None,
        "created_at": None, "updated_at": None,
    }
    row.update(overrides)
    return row


class IngestionSourceRepositoryTests(unittest.TestCase):
    def test_source_metadata_persists_and_is_user_scoped(self):
        connection = Mock()
        connection.execute.return_value.fetchone.return_value = source_row()
        with patch("database.repositories.get_connection", return_value=connection):
            created = create_ingestion_source(
                source_id="source-1", user_id="user-a",
                original_file_name="notes.txt", extension=".txt",
                media_type="text/plain", size_bytes=4, sha256="a" * 64,
                status="validated", memory_id="memory-1",
            )
            loaded = get_ingestion_source("source-1", "user-a")
        self.assertEqual(created.source_id, "source-1")
        self.assertEqual(loaded.user_id, "user-a")
        query, params = connection.execute.call_args.args
        self.assertIn("WHERE source_id = ? AND user_id = ?", query)
        self.assertEqual(params, ("source-1", "user-a"))
        self.assertGreaterEqual(connection.commit.call_count, 1)
        self.assertGreaterEqual(connection.close.call_count, 1)

    def test_duplicate_detection_only_returns_loaded_same_user_memory(self):
        connection = Mock()
        connection.execute.return_value.fetchone.return_value = {"memory_id": "memory-1"}
        with patch("database.repositories.get_connection", return_value=connection):
            duplicate = find_duplicate_content("user-a", "b" * 64)
        query, params = connection.execute.call_args.args
        self.assertIn("user_id = ?", query)
        self.assertIn("status = 'loaded'", query)
        self.assertEqual(params, ("user-a", "b" * 64))
        self.assertEqual(duplicate, "memory-1")

    def test_status_update_is_allowlisted_and_user_scoped(self):
        connection = Mock()
        connection.execute.return_value.rowcount = 1
        with patch("database.repositories.get_connection", return_value=connection):
            self.assertTrue(update_ingestion_source(
                "source-1", "user-a", status="failed", error_category="validation"
            ))
            with self.assertRaises(ValueError):
                update_ingestion_source("source-1", "user-a", sha256="d" * 64)
        query, params = connection.execute.call_args.args
        self.assertIn("WHERE source_id = ? AND user_id = ?", query)
        self.assertEqual(params[-2:], ("source-1", "user-a"))

    def test_zero_byte_failed_attempt_can_be_recorded(self):
        connection = Mock()
        connection.execute.return_value.fetchone.return_value = source_row(
            size_bytes=0, status="failed", memory_id=None
        )
        with patch("database.repositories.get_connection", return_value=connection):
            source = create_ingestion_source(
                source_id="empty-source", user_id="user-a",
                original_file_name="empty.txt", extension=".txt",
                media_type="text/plain", size_bytes=0, sha256="c" * 64,
                status="failed",
            )
        self.assertEqual(source.size_bytes, 0)
        self.assertEqual(source.status, "failed")


if __name__ == "__main__":
    unittest.main()
