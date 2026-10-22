import unittest

from memory.memory_models import (
    InvalidStateTransition,
    JobStatus,
    MemoryChunk,
    MemoryStatus,
    PipelineStage,
    PreparationErrorCategory,
    SourceDocument,
    StageResult,
    validate_job_transition,
    validate_memory_transition,
)


class LifecycleContractTests(unittest.TestCase):
    def test_valid_job_transitions(self):
        validate_job_transition(JobStatus.QUEUED, JobStatus.RUNNING)
        validate_job_transition(JobStatus.RUNNING, JobStatus.RETRYING)
        validate_job_transition(JobStatus.RETRYING, JobStatus.RUNNING)
        validate_job_transition(JobStatus.RUNNING, JobStatus.SUCCEEDED)
        validate_job_transition(JobStatus.FAILED, JobStatus.RETRYING)

    def test_invalid_job_transition_is_rejected(self):
        with self.assertRaises(InvalidStateTransition):
            validate_job_transition(JobStatus.QUEUED, JobStatus.SUCCEEDED)
        with self.assertRaises(InvalidStateTransition):
            validate_job_transition(JobStatus.SUCCEEDED, JobStatus.RUNNING)

    def test_memory_transitions(self):
        validate_memory_transition(MemoryStatus.PENDING, MemoryStatus.PROCESSING)
        validate_memory_transition(MemoryStatus.PROCESSING, MemoryStatus.READY)
        validate_memory_transition(MemoryStatus.FAILED, MemoryStatus.PROCESSING)
        with self.assertRaises(InvalidStateTransition):
            validate_memory_transition(MemoryStatus.PENDING, MemoryStatus.READY)

    def test_source_document_validation(self):
        doc = SourceDocument(
            source_id="src-1", memory_id="mem-1", file_name="notes.txt",
            media_type="text/plain", text="hello", metadata={"page": 1},
        )
        self.assertEqual(doc.extraction_status, "succeeded")
        with self.assertRaises(ValueError):
            SourceDocument(
                source_id="src-1", memory_id="mem-1", file_name="empty.txt",
                media_type="text/plain", text="   ",
            )

    def test_chunk_validation(self):
        chunk = MemoryChunk(
            chunk_id="chunk-1", memory_id="mem-1", source_id="src-1",
            chunk_index=0, text="searchable content", metadata={"page": 2},
        )
        self.assertEqual(chunk.chunk_index, 0)
        with self.assertRaises(ValueError):
            MemoryChunk(
                chunk_id="chunk-2", memory_id="mem-1", source_id="src-1",
                chunk_index=-1, text="text",
            )

    def test_failed_stage_requires_category(self):
        with self.assertRaises(ValueError):
            StageResult(stage=PipelineStage.EXTRACTION, succeeded=False)
        result = StageResult(
            stage=PipelineStage.EXTRACTION,
            succeeded=False,
            error_category=PreparationErrorCategory.EXTRACTION,
            error_message="Text extraction failed",
        )
        self.assertFalse(result.succeeded)


if __name__ == "__main__":
    unittest.main()
