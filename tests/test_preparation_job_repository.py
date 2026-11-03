"""PostgreSQL preparation-job repository contracts using an isolated fake connection."""
import unittest
from unittest.mock import patch

from database.repositories import (
    create_preparation_job, get_preparation_job, list_preparation_jobs,
    transition_preparation_job, update_preparation_job_progress,
)
from memory.memory_models import (
    InvalidStateTransition, JobStatus, PipelineStage, PreparationErrorCategory,
)


class _Cursor:
    def __init__(self, row=None, rows=None, rowcount=1):
        self.row = row
        self.rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(self):
        self.memory_ids = {("memory-1", "user-1"), ("memory-2", "user-1")}
        self.job = None
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def execute(self, query, params=None):
        params = params or ()
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT id FROM memories"):
            exists = (params[0], params[1]) in self.memory_ids
            return _Cursor({"id": params[0]} if exists else None)
        if normalized.startswith("INSERT INTO preparation_jobs"):
            job_id, memory_id, user_id, status, stage, total = params
            self.job = {
                "id": job_id, "memory_id": memory_id, "user_id": user_id,
                "status": status, "current_stage": stage, "attempt_count": 0,
                "completed_files": 0, "total_files": total,
                "error_category": None, "error_message": None,
                "created_at": None, "updated_at": None,
                "started_at": None, "finished_at": None,
            }
            return _Cursor()
        if normalized.startswith("SELECT * FROM preparation_jobs"):
            if self.job is None or (params[0], params[1]) != (self.job["id"], self.job["user_id"]):
                return _Cursor(None)
            if "WHERE user_id = ?" in normalized:
                return _Cursor(rows=[dict(self.job)])
            return _Cursor(dict(self.job))
        if normalized.startswith("UPDATE preparation_jobs"):
            if self.job is None:
                return _Cursor(rowcount=0)
            if "attempt_count = attempt_count + 1" in normalized:
                status, stage, job_id, user_id = params
                if (job_id, user_id) == (self.job["id"], self.job["user_id"]):
                    self.job.update(
                        status=status, current_stage=stage,
                        attempt_count=self.job["attempt_count"] + 1,
                        completed_files=0, error_category=None, error_message=None,
                    )
            elif "SET current_stage = ?" in normalized:
                stage, completed, total, job_id, user_id = params
                if (job_id, user_id) == (self.job["id"], self.job["user_id"]):
                    self.job.update(current_stage=stage, completed_files=completed, total_files=total)
            else:
                status = params[0]
                if status in {JobStatus.RETRYING.value, JobStatus.FAILED.value}:
                    _, category, message, job_id, user_id = params
                    self.job.update(status=status, error_category=category, error_message=message)
                elif status == JobStatus.SUCCEEDED.value:
                    _, stage, job_id, user_id = params
                    self.job.update(status=status, current_stage=stage)
            return _Cursor(rowcount=1)
        return _Cursor()

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class PreparationJobRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.connection = _FakeConnection()
        patcher = patch("database.repositories.get_connection", return_value=self.connection)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_job_lifecycle_and_progress_persist(self):
        job = create_preparation_job("memory-1", "user-1", job_id="job-1")
        self.assertEqual(job.status, JobStatus.QUEUED)
        self.assertEqual(job.current_stage, PipelineStage.DISCOVERY)
        job = transition_preparation_job("job-1", "user-1", JobStatus.RUNNING)
        self.assertEqual(job.attempt_count, 1)
        for stage in (
            PipelineStage.DISCOVERY, PipelineStage.EXTRACTION,
            PipelineStage.CHUNKING, PipelineStage.EMBEDDING, PipelineStage.INDEXING,
        ):
            job = update_preparation_job_progress(
                "job-1", "user-1", stage,
                completed_files=1 if stage == PipelineStage.INDEXING else 0,
                total_files=1,
            )
        job = transition_preparation_job("job-1", "user-1", JobStatus.SUCCEEDED)
        self.assertEqual(job.status, JobStatus.SUCCEEDED)
        self.assertEqual(job.current_stage, PipelineStage.COMPLETE)
        self.assertEqual(get_preparation_job("job-1", "user-1"), job)
        self.assertTrue(self.connection.committed)
        self.assertTrue(self.connection.closed)

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
        create_preparation_job("memory-1", "user-1", job_id="job-3")
        self.assertIsNone(get_preparation_job("job-3", "another-user"))
        self.assertEqual(list_preparation_jobs("another-user"), [])
        with self.assertRaises(LookupError):
            create_preparation_job("memory-1", "another-user", job_id="job-cross-user")


if __name__ == "__main__":
    unittest.main()
