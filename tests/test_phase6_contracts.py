"""Focused Phase 6 tests; run from the project root with unittest discovery."""
import unittest
from unittest.mock import patch

from memory.memory_models import (
    JobStatus,
    PipelineStage,
    PreparationErrorCategory,
    validate_job_transition,
    InvalidStateTransition,
)
from memory.background_worker import _classify_error


class Phase6ContractTests(unittest.TestCase):
    def test_queued_job_can_be_claimed(self):
        validate_job_transition(JobStatus.QUEUED, JobStatus.RUNNING)

    def test_running_job_can_retry_or_fail(self):
        validate_job_transition(JobStatus.RUNNING, JobStatus.RETRYING)
        validate_job_transition(JobStatus.RUNNING, JobStatus.FAILED)

    def test_succeeded_job_cannot_be_restarted(self):
        with self.assertRaises(InvalidStateTransition):
            validate_job_transition(JobStatus.SUCCEEDED, JobStatus.RUNNING)

    def test_extraction_failure_is_categorized_without_provider_error_text(self):
        category, message = _classify_error(PipelineStage.EXTRACTION, RuntimeError("private provider payload"))
        self.assertEqual(category, PreparationErrorCategory.EXTRACTION)
        self.assertNotIn("private provider payload", message)

    def test_indexing_failure_is_categorized(self):
        category, _ = _classify_error(PipelineStage.INDEXING, RuntimeError("boom"))
        self.assertEqual(category, PreparationErrorCategory.INDEXING)


if __name__ == "__main__":
    unittest.main()
