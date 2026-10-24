"""Regression tests for bounded, parallel OCR in PDF extraction."""
import logging
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz

from config.settings import settings
from ingestion.document_processor import (
    _is_retryable_ocr_error,
    _ocr_pdf_page_with_retries,
    extract_pdf_content,
)


def _create_pdf(path: Path, pages: int) -> None:
    document = fitz.open()
    for _ in range(pages):
        document.new_page()
    document.save(str(path))
    document.close()


class _ProviderError(Exception):
    def __init__(self, status_code: int, headers=None):
        super().__init__(f"provider status {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


class ParallelPdfOcrTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.settings_patches = [
            patch.object(settings, "PDF_OCR_ENABLED", True),
            patch.object(settings, "PDF_OCR_MIN_PAGE_TEXT_CHARS", 20),
            patch.object(settings, "PDF_OCR_DPI", 100),
            patch.object(settings, "PDF_OCR_MAX_RETRIES", 2),
            patch.object(settings, "PDF_OCR_RETRY_BASE_DELAY_SECONDS", 0.01),
            patch.object(settings, "PDF_OCR_RETRY_MAX_DELAY_SECONDS", 0.05),
        ]
        for patcher in self.settings_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_out_of_order_provider_completion_keeps_document_page_order(self):
        path = self.root / "out-of-order.pdf"
        _create_pdf(path, 3)
        completed = []

        def fake_ocr(_image_path, page_number, _file_name):
            time.sleep(0.08 if page_number == 1 else 0.01)
            completed.append(page_number)
            return f"Recognized content for page {page_number}"

        with patch.object(settings, "PDF_OCR_WORKERS", 3), \
             patch("ingestion.document_processor._ocr_pdf_page", side_effect=fake_ocr):
            result = extract_pdf_content(str(path))

        self.assertLess(result.text.index("Recognized content for page 1"),
                        result.text.index("Recognized content for page 2"))
        self.assertLess(result.text.index("Recognized content for page 2"),
                        result.text.index("Recognized content for page 3"))
        self.assertEqual(result.metadata["page_count"], 3)
        self.assertEqual(result.metadata["pages_with_text"], 3)
        self.assertGreater(completed.index(1), completed.index(2))

    def test_provider_concurrency_never_exceeds_configured_workers(self):
        path = self.root / "bounded.pdf"
        _create_pdf(path, 6)
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def fake_ocr(_image_path, page_number, _file_name):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return f"Content page {page_number}"

        with patch.object(settings, "PDF_OCR_WORKERS", 2), \
             patch("ingestion.document_processor._ocr_pdf_page", side_effect=fake_ocr):
            result = extract_pdf_content(str(path))

        self.assertEqual(result.metadata["pages_with_text"], 6)
        self.assertGreater(maximum_active, 1)
        self.assertLessEqual(maximum_active, 2)

    def test_retry_after_header_is_respected_and_transient_failure_recovers(self):
        calls = []

        def flaky_ocr(_image_path, page_number, _file_name):
            calls.append(page_number)
            if len(calls) == 1:
                raise _ProviderError(429, {"Retry-After": "2"})
            return "Recovered OCR text"

        with patch("ingestion.document_processor._ocr_pdf_page", side_effect=flaky_ocr), \
             patch("ingestion.document_processor.time.sleep") as sleep, \
             patch("ingestion.document_processor.random.uniform", return_value=0.0):
            text, retries, error_name = _ocr_pdf_page_with_retries(
                "unused.png", 1, "sample.pdf", 2, 0.01, 0.05
            )

        self.assertEqual(text, "Recovered OCR text")
        self.assertEqual(retries, 1)
        self.assertIsNone(error_name)
        sleep.assert_called_once_with(2.0)
        self.assertEqual(len(calls), 2)

    def test_permanent_failure_is_not_retried_and_other_pages_survive(self):
        path = self.root / "partial-failure.pdf"
        _create_pdf(path, 2)
        calls = []

        def fake_ocr(_image_path, page_number, _file_name):
            calls.append(page_number)
            if page_number == 2:
                raise _ProviderError(400)
            return "Usable OCR content on page one"

        with patch.object(settings, "PDF_OCR_WORKERS", 2), \
             patch("ingestion.document_processor._ocr_pdf_page", side_effect=fake_ocr), \
             patch("ingestion.document_processor.time.sleep") as sleep:
            result = extract_pdf_content(str(path))

        self.assertIn("Usable OCR content on page one", result.text)
        self.assertNotIn("Usable OCR content on page two", result.text)
        self.assertEqual(calls.count(2), 1)
        sleep.assert_not_called()
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("OCR could not process PDF page 2" in warning for warning in result.warnings))

    def test_transient_retries_stop_at_configured_limit(self):
        path = self.root / "retry-exhaustion.pdf"
        _create_pdf(path, 2)
        calls = []

        def fake_ocr(_image_path, page_number, _file_name):
            calls.append(page_number)
            if page_number == 2:
                raise _ProviderError(503)
            return "Page one OCR result"

        with patch.object(settings, "PDF_OCR_WORKERS", 2), \
             patch("ingestion.document_processor._ocr_pdf_page", side_effect=fake_ocr), \
             patch("ingestion.document_processor.time.sleep"):
            result = extract_pdf_content(str(path))

        self.assertEqual(calls.count(2), 3)  # initial attempt plus two retries
        self.assertIn("Page one OCR result", result.text)
        self.assertEqual(result.status, "partial")

    def test_retry_classifier_rejects_permanent_http_errors(self):
        self.assertTrue(_is_retryable_ocr_error(_ProviderError(429)))
        self.assertTrue(_is_retryable_ocr_error(_ProviderError(503)))
        self.assertFalse(_is_retryable_ocr_error(_ProviderError(400)))
        self.assertFalse(_is_retryable_ocr_error(ValueError("invalid image input")))


if __name__ == "__main__":
    unittest.main()
