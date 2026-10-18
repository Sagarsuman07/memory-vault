import hashlib
import tempfile
import unittest
from pathlib import Path

from ingestion.discovery import (
    SourceValidationError,
    detect_source_change,
    discover_file,
    discover_upload,
    get_memory_type,
)


class DiscoveryTests(unittest.TestCase):
    def test_discovers_text_upload_and_hashes_bytes(self):
        payload = b"hello memory vault\n"
        with tempfile.TemporaryDirectory() as directory:
            descriptor = discover_upload("notes.TXT", payload, upload_dir=directory, source_id="source-1")
        self.assertEqual(descriptor.source_id, "source-1")
        self.assertEqual(descriptor.extension, ".txt")
        self.assertEqual(descriptor.media_type, "text/plain")
        self.assertEqual(descriptor.size_bytes, len(payload))
        self.assertEqual(descriptor.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(get_memory_type("notes.TXT"), "document")

    def test_rejects_unsupported_extension(self):
        with self.assertRaises(SourceValidationError):
            discover_upload("script.exe", b"MZ", upload_dir=".")

    def test_rejects_empty_and_oversized_files(self):
        with self.assertRaisesRegex(SourceValidationError, "empty"):
            discover_upload("empty.txt", b"", upload_dir=".")
        with self.assertRaisesRegex(SourceValidationError, "cannot exceed"):
            discover_upload("large.txt", b"12345", upload_dir=".", max_file_size=4)

    def test_rejects_invalid_pdf_signature(self):
        with self.assertRaisesRegex(SourceValidationError, "PDF signature"):
            discover_upload("broken.pdf", b"not a pdf", upload_dir=".")

    def test_rejects_path_traversal_filename(self):
        descriptor = discover_upload("../../notes.txt", b"text", upload_dir="/tmp", source_id="safe-id")
        self.assertEqual(descriptor.file_name, "notes.txt")
        self.assertEqual(Path(descriptor.file_path).name, "safe-id.txt")

    def test_detects_file_change_by_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("first version", encoding="utf-8")
            original = discover_file(path, source_id="source-2")
            path.write_text("second version", encoding="utf-8")
            result = detect_source_change(original.sha256, path, "source-2")
        self.assertTrue(result.changed)
        self.assertNotEqual(result.previous_sha256, result.current_sha256)

    def test_unchanged_file_is_not_reported_as_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("same version", encoding="utf-8")
            original = discover_file(path, source_id="source-3")
            result = detect_source_change(original.sha256, path, "source-3")
        self.assertFalse(result.changed)


if __name__ == "__main__":
    unittest.main()
