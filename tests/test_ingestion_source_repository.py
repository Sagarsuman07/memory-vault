import sqlite3
import tempfile
import unittest
from pathlib import Path

import database.database as database_module
from database.database import initialize_database
from database.repositories import (
    create_ingestion_source,
    find_duplicate_content,
    get_ingestion_source,
    update_ingestion_source,
)


class IngestionSourceRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_database_path = database_module.DATABASE_PATH
        database_module.DATABASE_PATH = Path(self.temp_dir.name) / "test.db"
        initialize_database()

    def tearDown(self):
        database_module.DATABASE_PATH = self.old_database_path
        self.temp_dir.cleanup()

    def create_source(self, *, source_id="source-1", user_id="user-a", memory_id=None,
                      sha256="a" * 64, status="validated", size_bytes=4):
        return create_ingestion_source(
            source_id=source_id,
            user_id=user_id,
            original_file_name="notes.txt",
            extension=".txt",
            media_type="text/plain",
            size_bytes=size_bytes,
            sha256=sha256,
            status=status,
            memory_id=memory_id,
        )

    def test_source_metadata_persists_and_is_user_scoped(self):
        self.create_source()
        self.assertIsNotNone(get_ingestion_source("source-1", "user-a"))
        self.assertIsNone(get_ingestion_source("source-1", "user-b"))

    def test_duplicate_detection_only_returns_loaded_same_user_memory(self):
        self.create_source(memory_id="memory-1", status="loaded", sha256="b" * 64)
        self.assertEqual(find_duplicate_content("user-a", "b" * 64), "memory-1")
        self.assertIsNone(find_duplicate_content("user-b", "b" * 64))

    def test_failed_or_unloaded_sources_are_not_duplicate_candidates(self):
        self.create_source(memory_id=None, status="failed", sha256="c" * 64)
        self.assertIsNone(find_duplicate_content("user-a", "c" * 64))

    def test_status_update_is_allowlisted_and_user_scoped(self):
        self.create_source()
        self.assertTrue(update_ingestion_source("source-1", "user-a", status="failed", error_category="validation"))
        self.assertEqual(get_ingestion_source("source-1", "user-a").status, "failed")
        self.assertFalse(update_ingestion_source("source-1", "user-b", status="loaded"))
        with self.assertRaises(ValueError):
            update_ingestion_source("source-1", "user-a", sha256="d" * 64)

    def test_zero_byte_failed_attempt_can_be_recorded(self):
        source = self.create_source(source_id="empty-source", status="failed", size_bytes=0)
        self.assertEqual(source.size_bytes, 0)

    def test_phase2_schema_addition_preserves_existing_memory_rows(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        connection = sqlite3.connect(legacy_path)
        connection.execute("""CREATE TABLE memories (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, memory_type TEXT NOT NULL,
            title TEXT NOT NULL, file_name TEXT NOT NULL, file_path TEXT NOT NULL,
            summary TEXT, extracted_text TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )""")
        connection.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy-memory", "user-a", "document", "Old", "old.txt", "uploads/old.txt",
             None, "existing text", "2026-01-01", "2026-01-01"),
        )
        connection.commit()
        connection.close()

        database_module.DATABASE_PATH = legacy_path
        initialize_database()
        connection = sqlite3.connect(legacy_path)
        row = connection.execute(
            "SELECT id, extracted_text FROM memories WHERE id = ?", ("legacy-memory",)
        ).fetchone()
        tables = {item[0] for item in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()}
        connection.close()
        self.assertEqual(row, ("legacy-memory", "existing text"))
        self.assertIn("ingestion_sources", tables)
        self.assertIn("preparation_jobs", tables)


if __name__ == "__main__":
    unittest.main()
