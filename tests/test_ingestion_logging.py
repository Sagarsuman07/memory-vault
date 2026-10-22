"""Focused tests for safe, failure-tolerant ingestion logging."""
import logging
import unittest
from unittest.mock import Mock

from config.logging_config import StageTimer, _format_value, log_event


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
