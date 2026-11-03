"""Tests for summary extraction text retrieval from R2 and legacy records."""

import unittest
from unittest.mock import patch

from memory import memory_service


class ExtractedTextRetrievalTests(unittest.TestCase):
    def test_reads_extracted_text_from_object_storage(self):
        memory = {
            "extracted_text_key": "private/extracted/content.txt",
            "extracted_text": None,
        }
        with (
            patch.object(memory_service, "get_memory_by_id", return_value=memory),
            patch.object(
                memory_service,
                "download_text",
                return_value="Text extracted from the uploaded memory.",
            ) as download_text,
        ):
            result = memory_service.get_memory_extracted_text(
                "memory-1", "demo-user"
            )

        self.assertEqual(result, "Text extracted from the uploaded memory.")
        download_text.assert_called_once_with("private/extracted/content.txt")

    def test_falls_back_to_legacy_database_text(self):
        memory = {
            "extracted_text_key": None,
            "extracted_text": "Legacy extracted content.",
        }
        with (
            patch.object(memory_service, "get_memory_by_id", return_value=memory),
            patch.object(memory_service, "download_text") as download_text,
        ):
            result = memory_service.get_memory_extracted_text(
                "memory-1", "demo-user"
            )

        self.assertEqual(result, "Legacy extracted content.")
        download_text.assert_not_called()

    def test_returns_empty_text_when_no_extracted_content_exists(self):
        memory = {
            "extracted_text_key": None,
            "extracted_text": None,
        }
        with patch.object(memory_service, "get_memory_by_id", return_value=memory):
            result = memory_service.get_memory_extracted_text(
                "memory-1", "demo-user"
            )

        self.assertEqual(result, "")

    def test_rejects_non_text_storage_content(self):
        memory = {
            "extracted_text_key": "private/extracted/content.txt",
            "extracted_text": None,
        }
        with (
            patch.object(memory_service, "get_memory_by_id", return_value=memory),
            patch.object(memory_service, "download_text", return_value=b"not text"),
        ):
            with self.assertRaisesRegex(
                ValueError, "Stored extracted content is not valid text"
            ):
                memory_service.get_memory_extracted_text(
                    "memory-1", "demo-user"
                )


if __name__ == "__main__":
    unittest.main()
