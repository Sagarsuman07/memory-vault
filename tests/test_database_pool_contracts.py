"""Focused regression tests for PostgreSQL pool and job-lock contracts."""
import unittest
from unittest.mock import patch

from database.database import _PooledConnection
from database.repositories import transition_preparation_job
from memory.memory_models import JobStatus, PipelineStage


class _FakePool:
    def __init__(self):
        self.returned = []

    def putconn(self, connection):
        self.returned.append(connection)


class _FakeConnection:
    autocommit = False

    def __init__(self):
        self.rollback_count = 0

    def rollback(self):
        self.rollback_count += 1


class _FakeCursor:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class _JobConnection:
    def __init__(self):
        self.queries = []
        self.select_count = 0
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def execute(self, query, params=None):
        self.queries.append(query)
        if query.strip().startswith("SELECT * FROM preparation_jobs"):
            self.select_count += 1
            row = {
                "id": "job-1",
                "memory_id": "memory-1",
                "user_id": "user-1",
                "status": "queued" if self.select_count == 1 else "running",
                "current_stage": PipelineStage.DISCOVERY.value,
                "attempt_count": 0 if self.select_count == 1 else 1,
                "completed_files": 0,
                "total_files": 1,
                "error_category": None,
                "error_message": None,
                "created_at": None,
                "updated_at": None,
                "started_at": None,
                "finished_at": None,
            }
            return _FakeCursor(row)
        return _FakeCursor(None)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class DatabasePoolContractsTests(unittest.TestCase):
    def test_pooled_connection_close_rolls_back_and_returns_once(self):
        raw_connection = _FakeConnection()
        pool = _FakePool()
        connection = _PooledConnection(raw_connection, pool)

        connection.close()
        connection.close()

        self.assertEqual(raw_connection.rollback_count, 1)
        self.assertEqual(pool.returned, [raw_connection])

    def test_running_transition_locks_job_row_before_reading_status(self):
        connection = _JobConnection()
        with patch("database.repositories.get_connection", return_value=connection):
            job = transition_preparation_job("job-1", "user-1", JobStatus.RUNNING)

        self.assertEqual(job.status, JobStatus.RUNNING)
        self.assertTrue(connection.committed)
        self.assertTrue(connection.closed)
        locked_selects = [
            query for query in connection.queries
            if query.strip().startswith("SELECT * FROM preparation_jobs")
        ]
        self.assertEqual(len(locked_selects), 2)
        self.assertTrue(all("FOR UPDATE" in query.upper() for query in locked_selects))


if __name__ == "__main__":
    unittest.main()
