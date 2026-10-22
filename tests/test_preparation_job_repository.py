import tempfile
import unittest
from pathlib import Path

import database.database as database_module
from database.database import initialize_database
from database.models import Memory
from database.repositories import (
    create_memory,
    get_memory,
    create_preparation_job,
    get_preparation_job,
    list_preparation_jobs,
    transition_preparation_job,
    update_preparation_job_progress,
)
from memory.memory_models import (
    InvalidStateTransition,
    JobStatus,
    PipelineStage,
    PreparationErrorCategory,
)


class PreparationJobRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_database_path = database_module.DATABASE_PATH
        database_module.DATABASE_PATH = Path(self.temp_dir.name) / "test.db"
        initialize_database()
        self._create_memory("memory-1", "user-1")
        self._create_memory("memory-2", "user-1")
        self._create_memory("memory-3", "user-1")
        self._create_memory("memory-4", "user-1")

    def _create_memory(self, memory_id, user_id):
        create_memory(Memory(
            id=memory_id, user_id=user_id, memory_type="document",
            title="Test memory", file_name="test.txt", file_path="uploads/test.txt",
            summary=None, extracted_text="test content",
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
        ))

    def tearDown(self):
        database_module.DATABASE_PATH = self.original_database_path
        self.temp_dir.cleanup()

    def test_job_lifecycle_and_progress_persist(self):
        job = create_preparation_job("memory-1", "user-1", job_id="job-1")
        self.assertEqual(job.status, JobStatus.QUEUED)
        self.assertEqual(job.current_stage, PipelineStage.DISCOVERY)

        job = transition_preparation_job("job-1", "user-1", JobStatus.RUNNING)
        self.assertEqual(job.attempt_count, 1)

        for stage in (
            PipelineStage.DISCOVERY,
            PipelineStage.EXTRACTION,
            PipelineStage.CHUNKING,
            PipelineStage.EMBEDDING,
            PipelineStage.INDEXING,
        ):
            job = update_preparation_job_progress(
                "job-1", "user-1", stage,
                completed_files=1 if stage == PipelineStage.INDEXING else 0,
                total_files=1,
            )

        job = transition_preparation_job("job-1", "user-1", JobStatus.SUCCEEDED)
        self.assertEqual(job.status, JobStatus.SUCCEEDED)
        self.assertEqual(job.current_stage, PipelineStage.COMPLETE)
        self.assertIsNotNone(job.finished_at)
        self.assertEqual(get_preparation_job("job-1", "user-1"), job)

    def test_failed_job_can_retry_but_cannot_skip_states(self):
        create_preparation_job("memory-2", "user-1", job_id="job-2")
        with self.assertRaises(InvalidStateTransition):
            transition_preparation_job("job-2", "user-1", JobStatus.SUCCEEDED)

        transition_preparation_job("job-2", "user-1", JobStatus.RUNNING)
        failed = transition_preparation_job(
            "job-2", "user-1", JobStatus.FAILED,
            error_category=PreparationErrorCategory.EXTRACTION,
            error_message="Extraction failed",
        )
        self.assertEqual(failed.status, JobStatus.FAILED)

        retrying = transition_preparation_job(
            "job-2", "user-1", JobStatus.RETRYING,
            error_category=PreparationErrorCategory.EXTRACTION,
            error_message="Temporary extraction failure",
        )
        self.assertEqual(retrying.status, JobStatus.RETRYING)
        running = transition_preparation_job("job-2", "user-1", JobStatus.RUNNING)
        self.assertEqual(running.status, JobStatus.RUNNING)
        self.assertEqual(running.attempt_count, 2)
        self.assertEqual(running.current_stage, PipelineStage.DISCOVERY)

    def test_jobs_are_scoped_to_user(self):
        create_preparation_job("memory-3", "user-1", job_id="job-3")
        self.assertIsNone(get_preparation_job("job-3", "another-user"))
        self.assertEqual(list_preparation_jobs("another-user"), [])
        with self.assertRaises(LookupError):
            create_preparation_job("memory-3", "another-user", job_id="job-cross-user")

    def test_existing_memory_crud_schema_remains_compatible(self):
        memory = get_memory("memory-1", "user-1")
        self.assertIsNotNone(memory)
        self.assertEqual(memory["extracted_text"], "test content")

    def test_database_initialization_is_idempotent_and_preserves_rows(self):
        create_preparation_job("memory-1", "user-1", job_id="job-preserve")
        initialize_database()
        self.assertIsNotNone(get_memory("memory-1", "user-1"))
        self.assertIsNotNone(get_preparation_job("job-preserve", "user-1"))

    def test_stage_cannot_skip_or_move_backwards(self):
        create_preparation_job("memory-4", "user-1", job_id="job-4")
        transition_preparation_job("job-4", "user-1", JobStatus.RUNNING)
        with self.assertRaises(InvalidStateTransition):
            update_preparation_job_progress("job-4", "user-1", PipelineStage.CHUNKING)
        update_preparation_job_progress("job-4", "user-1", PipelineStage.EXTRACTION)
        with self.assertRaises(InvalidStateTransition):
            update_preparation_job_progress("job-4", "user-1", PipelineStage.DISCOVERY)


if __name__ == "__main__":
    unittest.main()
