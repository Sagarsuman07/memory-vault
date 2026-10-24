"""Focused tests for safe, failure-tolerant ingestion logging."""
import logging
import unittest
from threading import Lock
from unittest.mock import Mock, patch

from config.logging_config import StageTimer, _format_value, log_context, log_event, timed_lock


class IngestionLoggingTests(unittest.TestCase):
    def test_event_values_are_single_line_and_bounded(self):
        value = _format_value(" first line\nsecond line ")
        self.assertEqual(value, "first_line_second_line")
        self.assertEqual(len(_format_value("x" * 250)), 200)

    def test_boolean_and_float_values_have_stable_format(self):
        self.assertEqual(_format_value(True), "True")
        self.assertEqual(_format_value(1.23456), "1.235")

    def test_log_event_omits_none_values(self):
        logger = Mock()
        log_event(logger, "INGEST_TEST", memory="memory-1", absent=None)
        logger.log.assert_called_once_with(
            logging.INFO, "%s %s", "INGEST_TEST", "memory=memory-1"
        )

    def test_log_event_does_not_raise_when_logger_fails(self):
        logger = Mock()
        logger.log.side_effect = OSError("simulated logging failure")
        log_event(logger, "INGEST_TEST", value="safe")
        logger.log.assert_called_once()

    def test_stage_timer_logs_success_duration(self):
        logger = Mock()
        with StageTimer(logger, "unit_test", memory="memory-1") as timer:
            timer.set(items=2)
        logger.log.assert_called_once()
        args = logger.log.call_args.args
        self.assertEqual(args[0], logging.INFO)
        self.assertIn("INGEST_TIMING", args[2])
        self.assertIn("stage=unit_test", args[3])
        self.assertNotIn("status=", args[3])

    def test_stage_timer_logs_exception_type_and_propagates(self):
        logger = Mock()
        with self.assertRaisesRegex(ValueError, "expected test failure"):
            with StageTimer(logger, "unit_test", memory="memory-1"):
                raise ValueError("expected test failure")
        logger.log.assert_called_once()
        args = logger.log.call_args.args
        self.assertEqual(args[0], logging.WARNING)
        self.assertIn("INGEST_TIMING", args[2])
        self.assertIn("status=error", args[3])
        self.assertIn("error=ValueError", args[3])

    def test_timed_lock_logs_wait_and_hold_durations(self):
        logger = Mock()
        with timed_lock(logger, Lock(), "test_lock", "test_operation", memory="memory-1"):
            pass
        logger.log.assert_called_once()
        args = logger.log.call_args.args
        self.assertIn("INGEST_LOCK_TIMING", args[2])
        self.assertIn("lock=test_lock", args[3])
        self.assertIn("operation=test_operation", args[3])
        self.assertIn("wait_seconds=", args[3])
        self.assertIn("hold_seconds=", args[3])

    def test_timed_lock_releases_lock_after_exception(self):
        logger = Mock()
        lock = Lock()
        with self.assertRaisesRegex(ValueError, "lock body failed"):
            with timed_lock(logger, lock, "test_lock", "test_operation"):
                raise ValueError("lock body failed")
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()
        self.assertIn("status=error", logger.log.call_args.args[3])

    def test_log_context_adds_correlation_fields_without_overriding_explicit_fields(self):
        logger = Mock()
        with log_context(job="job-1", user="user-1"):
            log_event(logger, "INGEST_CONTEXT_TEST", memory="memory-1", job="explicit-job")
        args = logger.log.call_args.args
        self.assertIn("job=explicit-job", args[3])
        self.assertIn("user=user-1", args[3])
        self.assertIn("memory=memory-1", args[3])


class QueueWaitLoggingTests(unittest.TestCase):
    def test_queue_wait_seconds_uses_persisted_timestamp(self):
        from datetime import datetime, timedelta, timezone
        from memory.background_worker import _queue_wait_seconds

        created_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        seconds = _queue_wait_seconds(created_at)
        self.assertIsNotNone(seconds)
        self.assertGreaterEqual(seconds, 4.0)
        self.assertLess(seconds, 10.0)

    def test_queue_wait_seconds_ignores_invalid_timestamp(self):
        from memory.background_worker import _queue_wait_seconds

        self.assertIsNone(_queue_wait_seconds("not-a-timestamp"))


class PdfTimingLoggingTests(unittest.TestCase):
    def test_pdf_ocr_summary_counts_attempts_and_preserves_page_order(self):
        import tempfile
        from pathlib import Path
        import fitz
        from ingestion.document_processor import extract_pdf_content
        from config.settings import settings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two-pages.pdf"
            document = fitz.open()
            page1 = document.new_page()
            page1.insert_text((72, 72), "This is page one with enough searchable text.")
            document.new_page()
            document.save(str(path))
            document.close()

            with patch.object(settings, "PDF_OCR_ENABLED", True), \
                 patch.object(settings, "PDF_OCR_MIN_PAGE_TEXT_CHARS", 20), \
                 patch("ingestion.document_processor._ocr_pdf_page", return_value="OCR text from page two"), \
                 patch("ingestion.document_processor.log_event") as event:
                result = extract_pdf_content(str(path))

        self.assertLess(result.text.index("page one"), result.text.index("OCR text from page two"))
        summary = next(call for call in event.call_args_list
                       if call.args[1] == "INGEST_PDF_OCR_SUMMARY")
        self.assertEqual(summary.kwargs["eligible"], 1)
        self.assertEqual(summary.kwargs["attempted"], 1)
        self.assertEqual(summary.kwargs["succeeded"], 1)
        self.assertEqual(summary.kwargs["failed"], 0)
        self.assertEqual(summary.kwargs["retried"], 0)


class ProviderLoggingTests(unittest.TestCase):
    def test_image_provider_timing_is_logged_without_changing_result(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from PIL import Image
        from ingestion.image_processor import extract_image_content

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (2, 2), "white").save(path)
            fake_client = SimpleNamespace(chat=lambda **kwargs: SimpleNamespace(
                message=SimpleNamespace(content="A white square")
            ))
            with patch("ingestion.image_processor.get_ollama_client", return_value=fake_client), \
                 patch("ingestion.image_processor.log_event") as event:
                result = extract_image_content(str(path))

        self.assertEqual(result.text, "A white square")
        names = [call.args[1] for call in event.call_args_list]
        self.assertIn("INGEST_TIMING", names)
        self.assertIn("INGEST_IMAGE_EXTRACTION_SUCCESS", names)

    def test_audio_provider_timing_is_logged_without_changing_result(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from ingestion.audio_processor import extract_audio_content

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            path.write_bytes(b"test audio bytes")
            fake_client = SimpleNamespace(audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
                    text="Meeting at three PM"
                ))
            ))
            with patch("ingestion.audio_processor.get_groq_client", return_value=fake_client), \
                 patch("ingestion.audio_processor.log_event") as event:
                result = extract_audio_content(str(path))

        self.assertEqual(result.text, "Meeting at three PM")
        names = [call.args[1] for call in event.call_args_list]
        self.assertIn("INGEST_TIMING", names)
        self.assertIn("INGEST_AUDIO_EXTRACTION_SUCCESS", names)


if __name__ == "__main__":
    unittest.main()
