"""PostgreSQL schema contract tests; no live Supabase connection required."""
import unittest
from unittest.mock import patch

from database import database as database_module


class _FakeConnection:
    def __init__(self):
        self.statements = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def execute(self, statement, params=None):
        self.statements.append(statement)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class PostgresSchemaContractTests(unittest.TestCase):
    def test_fresh_schema_contains_cloud_storage_metadata(self):
        connection = _FakeConnection()
        with patch.object(database_module, "_schema_initialized", False):
            with patch("database.database.get_connection", return_value=connection):
                database_module.initialize_database()
        schema = "\n".join(connection.statements)
        self.assertIn("CREATE TABLE IF NOT EXISTS memories", schema)
        self.assertIn("original_file_key TEXT", schema)
        self.assertIn("extracted_text_key TEXT", schema)
        self.assertIn("CREATE TABLE IF NOT EXISTS ingestion_sources", schema)
        self.assertIn("metadata_json TEXT NOT NULL DEFAULT '{}'", schema)
        self.assertTrue(connection.committed)
        self.assertFalse(connection.rolled_back)
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
