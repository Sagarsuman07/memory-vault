"""Persistent, restart-recoverable preparation worker for Memory Vault.

One daemon thread per Python process polls SQLite. Job claiming uses the existing
atomic RUNNING transition (BEGIN IMMEDIATE), so concurrent pollers cannot execute
the same job. Long-running jobs are not stolen from another live worker; only
RUNNING jobs older than the configured stale threshold are recovered.
"""
from __future__ import annotations

from pathlib import Path
import threading
import time
import traceback

from config.settings import settings
from database.database import get_connection
from database.repositories import (
    get_preparation_job,
    transition_preparation_job,
    update_preparation_job_progress,
    update_ingestion_source,
)
from memory.memory_models import JobStatus, PipelineStage, PreparationErrorCategory

_POLL_SECONDS = 1.0
_STALE_RUNNING_SECONDS = 30 * 60
_START_LOCK = threading.Lock()
_WORKER_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()


def _memory_row(memory_id: str, user_id: str):
    connection = get_connection()
    try:
        return connection.execute(
            "SELECT * FROM memories WHERE id = ? AND user_id = ?", (memory_id, user_id)
        ).fetchone()
    finally:
        connection.close()


def _save_extracted_text(memory_id: str, user_id: str, text: str) -> None:
    connection = get_connection()
    try:
        cursor = connection.execute(
            "UPDATE memories SET extracted_text = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
            (text, memory_id, user_id),
        )
        if cursor.rowcount != 1:
            raise LookupError("Memory record was not found while saving extracted content.")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _classify_error(stage: PipelineStage, error: Exception) -> tuple[PreparationErrorCategory, str]:
    if stage == PipelineStage.DISCOVERY:
        category = PreparationErrorCategory.VALIDATION
        message = "The original source file is unavailable or failed validation."
    elif stage == PipelineStage.EXTRACTION:
        category = PreparationErrorCategory.EXTRACTION
        message = "Source extraction failed. Check the configured extraction provider and source file."
    elif stage == PipelineStage.CHUNKING:
        category = PreparationErrorCategory.CHUNKING
        message = "Extracted content could not be prepared for indexing."
    elif stage == PipelineStage.EMBEDDING:
        category = PreparationErrorCategory.EMBEDDING
        message = "Embedding generation failed. Check the configured embedding model and dependencies."
    elif stage == PipelineStage.INDEXING:
        category = PreparationErrorCategory.INDEXING
        message = "Memory indexing or index verification failed."
    else:
        category = PreparationErrorCategory.INTERNAL
        message = "An unexpected preparation error occurred."
    # Keep errors safe and deterministic; do not persist provider exception text,
    # which can contain request details or other sensitive information.
    return category, message


def _advance(job_id: str, user_id: str, stage: PipelineStage, *, completed: int | None = None) -> None:
    update_preparation_job_progress(
        job_id, user_id, stage, completed_files=completed, total_files=1
    )


def process_preparation_job(job_id: str, user_id: str) -> bool:
    """Run one queued/retrying job; return True only if it reaches succeeded."""
    max_attempts = max(1, int(getattr(settings, "INGESTION_MAX_ATTEMPTS", 3)))
    retry_delay = max(0.0, float(getattr(settings, "INGESTION_RETRY_DELAY_SECONDS", 5)))

    while True:
        job = get_preparation_job(job_id, user_id)
        if job is None:
            return False
        if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
            return job.status == JobStatus.SUCCEEDED
        if job.status not in {JobStatus.QUEUED, JobStatus.RETRYING}:
            # Another worker owns a running job.
            return False
        try:
            transition_preparation_job(job_id, user_id, JobStatus.RUNNING)
        except Exception:
            # Another worker may have claimed it between read and transition.
            return False

        stage = PipelineStage.DISCOVERY
        try:
            row = _memory_row(job.memory_id, user_id)
            if row is None:
                raise FileNotFoundError("Memory record is missing.")
            source_path = Path(row["file_path"])
            if not source_path.is_file():
                raise FileNotFoundError("Stored source file is missing.")

            _advance(job_id, user_id, PipelineStage.EXTRACTION)
            stage = PipelineStage.EXTRACTION
            from ingestion.processing import process_content
            processed = process_content(source_path)
            if not processed.text.strip():
                raise ValueError("No usable text was extracted.")
            _save_extracted_text(job.memory_id, user_id, processed.text)
            update_ingestion_source(
                job.memory_id, user_id,
                metadata_json=__import__("json").dumps(dict(processed.metadata), ensure_ascii=False),
                warnings_json=__import__("json").dumps(list(processed.warnings), ensure_ascii=False),
            )

            _advance(job_id, user_id, PipelineStage.CHUNKING)
            stage = PipelineStage.CHUNKING
            from rag.chunker import split_text
            chunks = split_text(processed.text, metadata=processed.metadata)
            if not chunks:
                raise ValueError("No chunks could be created for indexing.")

            _advance(job_id, user_id, PipelineStage.EMBEDDING)
            stage = PipelineStage.EMBEDDING
            from rag.vector_store import index_memory, verify_memory_index

            def index_progress(value: str) -> None:
                nonlocal stage
                if value == "embedding":
                    stage = PipelineStage.EMBEDDING
                elif value == "indexing":
                    # Advance only once; progress updates are persistent.
                    current = get_preparation_job(job_id, user_id)
                    if current and current.current_stage == PipelineStage.EMBEDDING:
                        _advance(job_id, user_id, PipelineStage.INDEXING)
                    stage = PipelineStage.INDEXING

            index_memory(
                memory_id=job.memory_id,
                user_id=user_id,
                memory_type=row["memory_type"],
                chunks=chunks,
                progress_callback=index_progress,
            )
            current = get_preparation_job(job_id, user_id)
            if current and current.current_stage == PipelineStage.EMBEDDING:
                _advance(job_id, user_id, PipelineStage.INDEXING)
            stage = PipelineStage.INDEXING
            verification = verify_memory_index(job.memory_id, expected_chunks=chunks)
            if not verification.get("valid"):
                raise RuntimeError("The vector index did not pass post-write verification.")
            if verification.get("record_count") != len(chunks):
                raise RuntimeError("The verified vector count does not match the chunk count.")

            _advance(job_id, user_id, PipelineStage.INDEXING, completed=1)
            transition_preparation_job(job_id, user_id, JobStatus.SUCCEEDED)
            update_ingestion_source(
                job.memory_id, user_id, status="loaded", memory_id=job.memory_id,
                stored_file_path=str(source_path), error_category=None, error_message=None,
            )
            return True
        except Exception as error:
            category, safe_message = _classify_error(stage, error)
            latest = get_preparation_job(job_id, user_id)
            if latest is None or latest.status != JobStatus.RUNNING:
                return False
            if latest.attempt_count < max_attempts:
                try:
                    transition_preparation_job(
                        job_id, user_id, JobStatus.RETRYING,
                        error_category=category, error_message=safe_message,
                    )
                except Exception:
                    return False
                if retry_delay:
                    time.sleep(retry_delay)
                continue
            try:
                transition_preparation_job(
                    job_id, user_id, JobStatus.FAILED,
                    error_category=category, error_message=safe_message,
                )
                update_ingestion_source(
                    job.memory_id, user_id, status="failed",
                    error_category=category.value, error_message=safe_message,
                )
            except Exception:
                pass
            return False


def _stale_running_job_ids() -> list[tuple[str, str]]:
    """Find jobs abandoned by a crashed process; leave recent RUNNING jobs alone."""
    connection = get_connection()
    try:
        rows = connection.execute(
            """SELECT id, user_id FROM preparation_jobs
               WHERE status = 'running'
                 AND started_at IS NOT NULL
                 AND started_at <= datetime('now', ?)""",
            (f"-{_STALE_RUNNING_SECONDS} seconds",),
        ).fetchall()
        return [(row["id"], row["user_id"]) for row in rows]
    finally:
        connection.close()


def _recover_stale_jobs() -> None:
    for job_id, user_id in _stale_running_job_ids():
        job = get_preparation_job(job_id, user_id)
        if job is None or job.status != JobStatus.RUNNING:
            continue
        try:
            if job.attempt_count >= max(1, int(getattr(settings, "INGESTION_MAX_ATTEMPTS", 3))):
                transition_preparation_job(
                    job_id, user_id, JobStatus.FAILED,
                    error_category=PreparationErrorCategory.INTERNAL,
                    error_message="Preparation worker stopped before completion; retry limit reached.",
                )
            else:
                transition_preparation_job(
                    job_id, user_id, JobStatus.RETRYING,
                    error_category=PreparationErrorCategory.INTERNAL,
                    error_message="Preparation worker stopped before completion; job queued for recovery.",
                )
        except Exception:
            # A live worker may have advanced the job after the stale scan.
            continue


def run_worker_forever() -> None:
    # SQLite claims are safe across processes. This lightweight worker handles
    # one job at a time per process to limit simultaneous model calls.
    while not _STOP_EVENT.is_set():
        try:
            _recover_stale_jobs()
            connection = get_connection()
            try:
                pending = connection.execute(
                    "SELECT id, user_id FROM preparation_jobs "
                    "WHERE status IN ('queued', 'retrying') ORDER BY created_at, id"
                ).fetchall()
            finally:
                connection.close()
            for row in pending:
                if _STOP_EVENT.is_set():
                    break
                process_preparation_job(row["id"], row["user_id"])
        except Exception:
            # Worker must survive transient SQLite/provider errors. Diagnostics
            # intentionally omit exception payloads and uploaded content.
            pass
        _STOP_EVENT.wait(_POLL_SECONDS)


def start_background_worker() -> None:
    """Start the process-local daemon worker once; safe across Streamlit reruns."""
    global _WORKER_THREAD
    with _START_LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return
        _STOP_EVENT.clear()
        _WORKER_THREAD = threading.Thread(
            target=run_worker_forever, name="memory-vault-preparation-worker", daemon=True
        )
        _WORKER_THREAD.start()
