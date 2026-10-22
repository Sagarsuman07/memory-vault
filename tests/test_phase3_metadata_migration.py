import os
import sqlite3
import tempfile
import unittest
from pathlib import Path


class TestPhase3MetadataMigration(unittest.TestCase):
    def test_phase2_source_schema_gets_additive_metadata_column(self):
        # Test initialize_database against a temporary database, never a user's DB.
        from unittest.mock import patch
        from config import settings as settings_module
        from database import database as database_module

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "memory_vault.db"
            with patch.object(settings_module.settings, "DATABASE_PATH", str(db_path)):
                with patch.object(database_module, "DATABASE_PATH", db_path):
                    # Simulate an existing Phase 2 schema without metadata_json.
                    connection = sqlite3.connect(db_path)
                    connection.execute("""CREATE TABLE ingestion_sources (
                        source_id TEXT PRIMARY KEY, memory_id TEXT, user_id TEXT NOT NULL,
                        original_file_name TEXT NOT NULL, extension TEXT NOT NULL,
                        media_type TEXT NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
                        status TEXT NOT NULL, stored_file_path TEXT, duplicate_of_memory_id TEXT,
                        warnings_json TEXT NOT NULL DEFAULT '[]', error_category TEXT,
                        error_message TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
                    connection.execute("INSERT INTO ingestion_sources (source_id,user_id,original_file_name,extension,media_type,size_bytes,sha256,status) VALUES ('s1','u1','a.txt','.txt','text/plain',1,'abc','loaded')")
                    connection.commit()
                    connection.close()
                    database_module.initialize_database()
                    connection = sqlite3.connect(db_path)
                    columns = {row[1] for row in connection.execute("PRAGMA table_info(ingestion_sources)")}
                    row = connection.execute("SELECT source_id,metadata_json FROM ingestion_sources WHERE source_id='s1'").fetchone()
                    connection.close()
                    self.assertIn("metadata_json", columns)
                    self.assertEqual(row, ("s1", "{}"))

                    # Verify the new repository field round-trips as well.
                    from database.repositories import create_ingestion_source, get_ingestion_source
                    created = create_ingestion_source(
                        source_id="s2", user_id="u1", original_file_name="b.txt",
                        extension=".txt", media_type="text/plain", size_bytes=4,
                        sha256="b" * 64, status="loaded", memory_id="m2",
                        metadata_json='{"document_type":"text","line_count":1}',
                    )
                    loaded = get_ingestion_source("s2", "u1")
                    self.assertEqual(created.metadata_json, '{"document_type":"text","line_count":1}')
                    self.assertEqual(loaded.metadata_json, created.metadata_json)


if __name__ == "__main__":
    unittest.main()
