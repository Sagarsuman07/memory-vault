"""Unit tests for the private R2 storage adapter; no live credentials required."""
import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

import storage.file_store as file_store


class FileStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        client_patcher = patch.object(file_store, "_client", return_value=self.client)
        bucket_patcher = patch.object(file_store, "_bucket", return_value="memory-vault-test")
        client_patcher.start()
        bucket_patcher.start()
        self.addCleanup(client_patcher.stop)
        self.addCleanup(bucket_patcher.stop)

    def test_object_key_is_scoped_and_filename_is_sanitized(self):
        key = file_store.build_memory_object_key(
            "demo_user", "memory-1", "original", "../../receipt.pdf"
        )
        self.assertEqual(key, "memory-vault/users/demo_user/memories/memory-1/original/receipt.pdf")
        with self.assertRaises(ValueError):
            file_store.build_memory_object_key("../user", "memory-1", "original", "a.txt")

    def test_upload_file_rewinds_stream_before_upload(self):
        source = io.BytesIO(b"original bytes")
        source.seek(5)
        file_store.upload_file(source, "original/key", "text/plain")
        self.client.upload_fileobj.assert_called_once()
        self.assertEqual(self.client.upload_fileobj.call_args.args[0].tell(), 0)
        self.assertEqual(self.client.upload_fileobj.call_args.args[:3], (source, "memory-vault-test", "original/key"))

    def test_upload_text_writes_utf8_content(self):
        file_store.upload_text("Hello ₹", "extracted/key")
        self.client.put_object.assert_called_once_with(
            Bucket="memory-vault-test", Key="extracted/key",
            Body="Hello ₹".encode("utf-8"), ContentType="text/plain; charset=utf-8",
        )

    def test_download_to_temp_removes_temporary_file(self):
        def write_download(_bucket, _key, destination):
            Path(destination).write_bytes(b"stored file")
        self.client.download_file.side_effect = write_download
        with file_store.download_to_temp("original/key", suffix=".pdf") as path:
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), b"stored file")
        self.assertFalse(path.exists())

    def test_file_exists_returns_false_for_not_found(self):
        self.client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadObject"
        )
        self.assertFalse(file_store.file_exists("missing/key"))

    def test_file_exists_propagates_non_not_found_errors(self):
        self.client.head_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "HeadObject"
        )
        with self.assertRaises(ClientError):
            file_store.file_exists("private/key")

    def test_signed_url_uses_private_object_and_download_name(self):
        self.client.generate_presigned_url.return_value = "https://signed.example/file"
        result = file_store.generate_signed_url("original/key", expires_in=300, download_filename="my receipt.pdf")
        self.assertEqual(result, "https://signed.example/file")
        kwargs = self.client.generate_presigned_url.call_args.kwargs
        self.assertEqual(kwargs["ClientMethod"], "get_object")
        self.assertEqual(kwargs["Params"]["Bucket"], "memory-vault-test")
        self.assertEqual(kwargs["Params"]["Key"], "original/key")
        self.assertEqual(kwargs["ExpiresIn"], 300)
        self.assertIn("my%20receipt.pdf", kwargs["Params"]["ResponseContentDisposition"])

    def test_signed_url_rejects_unbounded_expiry(self):
        with self.assertRaises(ValueError):
            file_store.generate_signed_url("original/key", expires_in=7200)


if __name__ == "__main__":
    unittest.main()
