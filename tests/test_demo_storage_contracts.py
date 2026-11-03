"""Regression tests for deterministic demo memories stored in R2."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from database.models import Memory
from memory import memory_service


class DemoStorageTests(unittest.TestCase):
    def test_deterministic_demo_memory_stores_original_and_extracted_objects_in_r2(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "demo.txt"
            source.write_text("original demo source", encoding="utf-8")
            with (
                patch.object(memory_service, "upload_file") as upload_original,
                patch.object(memory_service, "upload_text") as upload_extracted,
                patch.object(memory_service, "create_memory") as create_memory,
                patch.object(memory_service, "create_ingestion_source") as create_source,
                patch.object(memory_service, "split_text", return_value=["chunk"]),
                patch.object(memory_service, "index_memory") as index_memory,
                patch.object(memory_service, "update_ingestion_source", return_value=True) as update_source,
            ):
                memory = memory_service._create_memory_from_demo_text(
                    title="[Demo] Text", file_path=source, memory_type="document",
                    extracted_text="normalized demo text", user_id="demo_user",
                )
        self.assertIsInstance(memory, Memory)
        self.assertTrue(memory.original_file_key.startswith("memory-vault/users/demo_user/"))
        self.assertTrue(memory.extracted_text_key.startswith("memory-vault/users/demo_user/"))
        self.assertIsNone(memory.extracted_text)
        upload_original.assert_called_once()
        upload_extracted.assert_called_once_with("normalized demo text", memory.extracted_text_key)
        create_memory.assert_called_once_with(memory)
        create_source.assert_called_once()
        self.assertEqual(create_source.call_args.kwargs["status"], "validated")
        index_memory.assert_called_once()
        update_source.assert_called_once()
        self.assertEqual(update_source.call_args.kwargs["status"], "loaded")

    def test_demo_source_files_are_temporary_not_persistent_uploads(self):
        created_source_paths = []

        def fake_create_memory(*, uploaded_file, title, user_id):
            created_source_paths.append(uploaded_file._path)
            return object()

        def fake_audio_memory(**kwargs):
            created_source_paths.append(kwargs["file_path"])
            return object()

        with (
            patch.object(memory_service, "list_memories", return_value=[]),
            patch.object(memory_service, "create_new_memory", side_effect=fake_create_memory),
            patch.object(memory_service, "_create_memory_from_demo_text", side_effect=fake_audio_memory),
        ):
            created = memory_service.load_demo_memories("demo_user")
        self.assertEqual(len(created), 5)
        self.assertEqual(len(created_source_paths), 5)
        self.assertTrue(all(not path.exists() for path in created_source_paths))


if __name__ == "__main__":
    unittest.main()
