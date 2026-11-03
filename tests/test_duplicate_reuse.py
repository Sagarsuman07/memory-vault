import unittest
from unittest.mock import Mock, patch

from memory.duplicate_reuse import _find_candidates


class DuplicateReuseCandidateTests(unittest.TestCase):
    @patch("memory.duplicate_reuse.get_connection")
    def test_candidate_query_uses_current_postgres_extracted_text_column(self, get_connection):
        connection = Mock()
        candidate_cursor = Mock()
        candidate_cursor.fetchall.return_value = [{
            "memory_id": "source-memory",
            "source_id": "source-file",
            "extension": ".pdf",
            "metadata_json": "{}",
            "warnings_json": "[]",
        }]
        count_cursor = Mock()
        count_cursor.fetchone.return_value = {"n": 1}
        connection.execute.side_effect = [candidate_cursor, count_cursor]
        get_connection.return_value = connection

        candidates, stale_count = _find_candidates(
            user_id="user-1",
            sha256="a" * 64,
            exclude_memory_id="target-memory",
        )

        candidate_sql = connection.execute.call_args_list[0].args[0]
        self.assertIn("m.legacy_extracted_text IS NOT NULL", candidate_sql)
        self.assertIn("m.extracted_text_key IS NOT NULL", candidate_sql)
        self.assertIn("m.legacy_extracted_text IS NOT NULL", candidate_sql)
        self.assertNotIn("m.extracted_text", candidate_sql)
        self.assertEqual(candidates[0]["memory_id"], "source-memory")
        self.assertEqual(stale_count, 0)
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
