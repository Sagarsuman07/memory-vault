"""Concurrent, persistent background preparation worker for Memory Vault.

The dispatcher runs one thread and submits individual memory jobs to a bounded
ThreadPoolExecutor. SQLite's atomic job transition remains the source of truth
for claiming work. The worker recovers RUNNING jobs left by a previous app
process once at startup; it does not periodically steal long-running jobs.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import json
import logging
import threading
import time

from config.settings import settings
from database.database import get_connection
from database.repositories import (
    get_preparation_job,
    transition_preparation_job,
    update_preparation_job_progress,
    update_ingestion_source,
    get_ingestion_source,
)
from memory.memory_models import JobStatus, PipelineStage, PreparationErrorCategory
from config.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_POLL_SECONDS = 0.5
_START_LOCK = threading.Lock()
_WORKER_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()


def _worker_count() -> int:
    """Read bounded concurrency from settings; at least one worker is required."""
    try:
        configured = int(getattr(settings, "INGESTION_WORKER_CONCURRENCY", 2))
    except (TypeError, ValueError):
        configured = 2
    return max(1, min(configured, 8))


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


def _classify_error(stage: PipelineStage) -> tuple[PreparationErrorCategory, str]:
    """Return a safe message; do not persist raw provider exceptions."""
    if stage == PipelineStage.DISCOVERY:
        return PreparationErrorCategory.VALIDATION, "The original source file is unavailable or failed validation."
    if stage == PipelineStage.EXTRACTION:
        return PreparationErrorCategory.EXTRACTION, "Source extraction failed. Check the configured extraction provider and source file."
    if stage == PipelineStage.CHUNKING:
        return PreparationErrorCategory.CHUNKING, "Extracted content could not be prepared for indexing."
    if stage == PipelineStage.EMBEDDING:
        return PreparationErrorCategory.EMBEDDING, "Embedding generation failed. Check the configured embedding model and dependencies."
    if stage == PipelineStage.INDEXING:
        return PreparationErrorCategory.INDEXING, "Memory indexing or index verification failed."
    return PreparationErrorCategory.INTERNAL, "An unexpected preparation error occurred."


def _advance(job_id: str, user_id: str, stage: PipelineStage, *, completed: int | None = None) -> None:
    update_preparation_job_progress(job_id, user_id, stage, completed_files=completed, total_files=1)


def process_preparation_job(job_id: str, user_id: str) -> bool:
    """Run one queued/retrying job; True means extraction and index verification succeeded."""
    max_attempts = max(1, int(getattr(settings, "INGESTION_MAX_ATTEMPTS", 3)))
    retry_delay = max(0.0, float(getattr(settings, "INGESTION_RETRY_DELAY_SECONDS", 5)))

    while True:
        job = get_preparation_job(job_id, user_id)
        if job is None:
            return False
        if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
            return job.status == JobStatus.SUCCEEDED
        if job.status not in {JobStatus.QUEUED, JobStatus.RETRYING}:
            # A different worker already claimed the job.
            return False
        try:
            transition_preparation_job(job_id, user_id, JobStatus.RUNNING)
        except Exception:
            # Atomic transition prevents duplicate execution if another process
            # claimed the same job between the read and the transition.
            return False

        stage = PipelineStage.DISCOVERY
        job_started = time.perf_counter()
        logger.info("INGEST_JOB_START job_id=%s memory_id=%s attempt=%s", job_id, job.memory_id, getattr(job, "attempt_count", None))
        try:
            row = _memory_row(job.memory_id, user_id)
            if row is None:
                raise FileNotFoundError("Memory record is missing.")
            source_path = Path(row["file_path"])
            if not source_path.is_file():
                raise FileNotFoundError("Stored source file is missing.")

            # Attempt duplicate reuse only when the upload was matched to a prior
            # successfully loaded source. Any failed eligibility check falls back
            # to the normal extraction/chunking/embedding pipeline.
            target_source = get_ingestion_source(job.memory_id, user_id)
            candidate_id = (target_source.duplicate_of_memory_id
                            if target_source is not None else None)
            logger.info(
                "INGEST_DUPLICATE_CHECK memory_id=%s candidate_memory_id=%s",
                job.memory_id, candidate_id,
            )
            reused = False
            verification = None
            expected_count = None
            duplicate_check_started = time.perf_counter()

            if candidate_id and candidate_id != job.memory_id:
                candidate_source = get_ingestion_source(candidate_id, user_id)
                candidate_row = _memory_row(candidate_id, user_id)
                # The hash lookup already limits candidates to loaded source rows.
                # Older successful memories may not have extracted_text persisted,
                # so do not reject an otherwise valid indexed duplicate for that reason.
                if candidate_source is None:
                    candidate_reason = "candidate_source_record_missing"
                elif candidate_source.status != "loaded":
                    candidate_reason = f"candidate_source_status_{candidate_source.status}"
                elif candidate_source.memory_id != candidate_id:
                    candidate_reason = "candidate_source_memory_id_mismatch"
                elif candidate_row is None:
                    candidate_reason = "candidate_memory_row_missing"
                else:
                    candidate_reason = None
                candidate_eligible = candidate_reason is None
                if not candidate_eligible:
                    logger.info(
                        "INGEST_DUPLICATE_FALLBACK memory_id=%s source_memory_id=%s reason=%s",
                        job.memory_id, candidate_id, candidate_reason,
                    )
                if candidate_eligible:
                    from rag.vector_store import (
                        clone_memory_index, verify_memory_index,
                    )
                    candidate_verification = verify_memory_index(candidate_id)
                    logger.info("INGEST_DUPLICATE_CANDIDATE memory_id=%s source_memory_id=%s source_index_valid=%s source_chunks=%s", job.memory_id, candidate_id, candidate_verification.get("valid"), candidate_verification.get("record_count"))
                    if not (candidate_verification.get("valid")
                            and candidate_verification.get("record_count", 0) > 0):
                        logger.info(
                            "INGEST_DUPLICATE_FALLBACK memory_id=%s source_memory_id=%s reason=source_index_invalid errors=%s",
                            job.memory_id, candidate_id,
                            ";".join(candidate_verification.get("errors", []))[:500],
                        )
                    if (candidate_verification.get("valid")
                            and candidate_verification.get("record_count", 0) > 0):
                        # Advance the persisted state machine in order while
                        # deliberately skipping the expensive work at each stage.
                        _advance(job_id, user_id, PipelineStage.EXTRACTION)
                        _advance(job_id, user_id, PipelineStage.CHUNKING)
                        _advance(job_id, user_id, PipelineStage.EMBEDDING)
                        _advance(job_id, user_id, PipelineStage.INDEXING)
                        stage = PipelineStage.INDEXING

                        source_text = candidate_row["extracted_text"]
                        if not isinstance(source_text, str) or not source_text.strip():
                            # Recover legacy/missing SQL extracted text from the
                            # already verified source chunks; this performs no OCR
                            # or embedding call and does not alter the source index.
                            from rag.vector_store import get_memory_index_text
                            source_text = get_memory_index_text(candidate_id, user_id)
                            if not source_text.strip():
                                raise RuntimeError(
                                    "Duplicate source has a valid index but no recoverable chunk text."
                                )
                            logger.info(
                                "INGEST_DUPLICATE_TEXT_RECOVERED memory_id=%s source_memory_id=%s chars=%d source=existing_index_chunks",
                                job.memory_id, candidate_id, len(source_text),
                            )
                        _save_extracted_text(job.memory_id, user_id, source_text)
                        clone_started = time.perf_counter()
                        expected_count = clone_memory_index(
                            source_memory_id=candidate_id,
                            target_memory_id=job.memory_id,
                            user_id=user_id,
                            memory_type=row["memory_type"],
                        )
                        logger.info("INGEST_TIMING stage=duplicate_vector_clone memory_id=%s source_memory_id=%s chunks=%s seconds=%.3f", job.memory_id, candidate_id, expected_count, time.perf_counter() - clone_started)
                        verification_started = time.perf_counter()
                        verification = verify_memory_index(job.memory_id)
                        logger.info("INGEST_TIMING stage=duplicate_target_verification memory_id=%s seconds=%.3f valid=%s records=%s", job.memory_id, time.perf_counter() - verification_started, verification.get("valid"), verification.get("record_count"))
                        if (not verification.get("valid")
                                or verification.get("record_count") != expected_count
                                or expected_count != candidate_verification.get("record_count")):
                            raise RuntimeError(
                                "Reused vector index did not pass post-write verification."
                            )
                        if candidate_source.metadata_json:
                            update_ingestion_source(
                                job.memory_id, user_id,
                                metadata_json=candidate_source.metadata_json,
                            )
                        reused = True
                        logger.info(
                            "INGEST_DUPLICATE_REUSE memory_id=%s source_memory_id=%s "
                            "chunks=%s extraction_skipped=true embedding_skipped=true",
                            job.memory_id, candidate_id, expected_count,
                        )

            if not candidate_id:
                logger.info(
                    "INGEST_DUPLICATE_FALLBACK memory_id=%s reason=no_completed_same_user_duplicate_found",
                    job.memory_id,
                )
            elif candidate_id == job.memory_id:
                logger.info(
                    "INGEST_DUPLICATE_FALLBACK memory_id=%s reason=candidate_is_same_memory",
                    job.memory_id,
                )
            logger.info("INGEST_TIMING stage=duplicate_check memory_id=%s seconds=%.3f reused=%s", job.memory_id, time.perf_counter() - duplicate_check_started, reused)
            if not reused:
                _advance(job_id, user_id, PipelineStage.EXTRACTION)
                stage = PipelineStage.EXTRACTION
                from ingestion.processing import process_content
                extraction_started = time.perf_counter()
                processed = process_content(source_path)
                logger.info("INGEST_TIMING stage=extraction memory_id=%s file=%s seconds=%.3f chars=%d", job.memory_id, source_path.name, time.perf_counter() - extraction_started, len(processed.text))
                if not processed.text.strip():
                    raise ValueError("No usable text was extracted.")
                _save_extracted_text(job.memory_id, user_id, processed.text)
                update_ingestion_source(
                    job.memory_id,
                    user_id,
                    metadata_json=json.dumps(dict(processed.metadata), ensure_ascii=False),
                    warnings_json=json.dumps(list(processed.warnings), ensure_ascii=False),
                )

                _advance(job_id, user_id, PipelineStage.CHUNKING)
                stage = PipelineStage.CHUNKING
                from rag.chunker import split_text
                chunking_started = time.perf_counter()
                chunks = split_text(processed.text, metadata=processed.metadata)
                logger.info("INGEST_TIMING stage=chunking memory_id=%s seconds=%.3f chars=%d chunks=%d", job.memory_id, time.perf_counter() - chunking_started, len(processed.text), len(chunks))
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
                        current = get_preparation_job(job_id, user_id)
                        if current and current.current_stage == PipelineStage.EMBEDDING:
                            _advance(job_id, user_id, PipelineStage.INDEXING)
                        stage = PipelineStage.INDEXING

                indexing_started = time.perf_counter()
                index_memory(
                    memory_id=job.memory_id,
                    user_id=user_id,
                    memory_type=row["memory_type"],
                    chunks=chunks,
                    progress_callback=index_progress,
                )
                logger.info("INGEST_TIMING stage=index_memory_total memory_id=%s seconds=%.3f chunks=%d", job.memory_id, time.perf_counter() - indexing_started, len(chunks))
                current = get_preparation_job(job_id, user_id)
                if current and current.current_stage == PipelineStage.EMBEDDING:
                    _advance(job_id, user_id, PipelineStage.INDEXING)
                stage = PipelineStage.INDEXING
                verification_started = time.perf_counter()
                verification = verify_memory_index(job.memory_id, expected_chunks=chunks)
                logger.info("INGEST_TIMING stage=final_index_verification memory_id=%s seconds=%.3f records=%s valid=%s", job.memory_id, time.perf_counter() - verification_started, verification.get("record_count"), verification.get("valid"))
                expected_count = len(chunks)
                if not verification.get("valid"):
                    raise RuntimeError("The vector index did not pass post-write verification.")
                if verification.get("record_count") != expected_count:
                    raise RuntimeError("The verified vector count does not match the chunk count.")

            _advance(job_id, user_id, PipelineStage.INDEXING, completed=1)
            transition_preparation_job(job_id, user_id, JobStatus.SUCCEEDED)
            logger.info("INGEST_JOB_SUCCESS job_id=%s memory_id=%s reused_duplicate=%s total_seconds=%.3f chunks=%s", job_id, job.memory_id, reused, time.perf_counter() - job_started, expected_count)
            update_ingestion_source(
                job.memory_id,
                user_id,
                status="loaded",
                memory_id=job.memory_id,
                stored_file_path=str(source_path),
                error_category=None,
                error_message=None,
            )
            return True

        except Exception:
            logger.exception("INGEST_JOB_ERROR job_id=%s memory_id=%s stage=%s elapsed_seconds=%.3f", job_id, getattr(job, "memory_id", None), stage, time.perf_counter() - job_started)
            category, safe_message = _classify_error(stage)
            latest = get_preparation_job(job_id, user_id)
            if latest is None or latest.status != JobStatus.RUNNING:
                return False
            if latest.attempt_count < max_attempts:
                try:
                    transition_preparation_job(
                        job_id,
                        user_id,
                        JobStatus.RETRYING,
                        error_category=category,
                        error_message=safe_message,
                    )
                except Exception:
                    return False
                if retry_delay:
                    time.sleep(retry_delay)
                continue
            try:
                transition_preparation_job(
                    job_id,
                    user_id,
                    JobStatus.FAILED,
                    error_category=category,
                    error_message=safe_message,
                )
                update_ingestion_source(
                    job.memory_id,
                    user_id,
                    status="failed",
                    error_category=category.value,
                    error_message=safe_message,
                )
            except Exception:
                pass
            return False


def _recover_interrupted_jobs_on_startup() -> None:
    """Requeue jobs left RUNNING by a previous app process.

    This runs once per worker startup, not on every poll. That avoids mistaking
    a legitimately long-running PDF for a dead worker and launching it twice.
    This local-app design assumes a single application process owns the worker.
    """
    connection = get_connection()
    try:
        rows = connection.execute(
            "SELECT id, user_id, attempt_count FROM preparation_jobs WHERE status = 'running'"
        ).fetchall()
        # Phase 7's Stop control has been retired. Clear only its obsolete request
        # records if that optional table exists; do not delete memories or vectors.
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'preparation_job_cancellations'"
        ).fetchone()
        if table:
            connection.execute("DELETE FROM preparation_job_cancellations")
            connection.commit()
    finally:
        connection.close()

    max_attempts = max(1, int(getattr(settings, "INGESTION_MAX_ATTEMPTS", 3)))
    for row in rows:
        try:
            job = get_preparation_job(row["id"], row["user_id"])
            if job is None or job.status != JobStatus.RUNNING:
                continue
            if job.attempt_count >= max_attempts:
                transition_preparation_job(
                    job.id,
                    job.user_id,
                    JobStatus.FAILED,
                    error_category=PreparationErrorCategory.INTERNAL,
                    error_message="Preparation was interrupted by an application restart; retry limit reached.",
                )
            else:
                transition_preparation_job(
                    job.id,
                    job.user_id,
                    JobStatus.RETRYING,
                    error_category=PreparationErrorCategory.INTERNAL,
                    error_message="Preparation was interrupted by an application restart; queued for recovery.",
                )
        except Exception:
            # Keep the dispatcher alive; a later startup or manual diagnosis can
            # handle a job whose state cannot be transitioned safely.
            continue


def _pending_jobs() -> list[tuple[str, str]]:
    connection = get_connection()
    try:
        rows = connection.execute(
            "SELECT id, user_id FROM preparation_jobs "
            "WHERE status IN ('queued', 'retrying') ORDER BY created_at, id"
        ).fetchall()
        return [(row["id"], row["user_id"]) for row in rows]
    finally:
        connection.close()


def run_worker_forever() -> None:
    """Dispatch jobs to a bounded pool so one large file cannot block all others."""
    concurrency = _worker_count()
    try:
        _recover_interrupted_jobs_on_startup()
    except Exception:
        # Startup recovery must not prevent fresh queued jobs from being served.
        pass
    in_flight: dict[Future, tuple[str, str]] = {}
    in_flight_keys: set[tuple[str, str]] = set()

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="memory-vault-job") as executor:
        while not _STOP_EVENT.is_set():
            # Drop finished futures and release their slots.
            for future, key in list(in_flight.items()):
                if future.done():
                    try:
                        future.result()
                    except Exception:
                        # Never log source contents or raw provider exceptions here.
                        pass
                    in_flight.pop(future, None)
                    in_flight_keys.discard(key)

            free_slots = concurrency - len(in_flight)
            if free_slots > 0:
                try:
                    for key in _pending_jobs():
                        if free_slots <= 0 or _STOP_EVENT.is_set():
                            break
                        if key in in_flight_keys:
                            continue
                        future = executor.submit(process_preparation_job, key[0], key[1])
                        in_flight[future] = key
                        in_flight_keys.add(key)
                        free_slots -= 1
                except Exception:
                    # Transient DB issues should not kill the dispatcher.
                    pass

            _STOP_EVENT.wait(_POLL_SECONDS)


def start_background_worker() -> None:
    """Start the process-local dispatcher once; safe across Streamlit reruns."""
    global _WORKER_THREAD
    with _START_LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return
        _STOP_EVENT.clear()
        _WORKER_THREAD = threading.Thread(
            target=run_worker_forever,
            name="memory-vault-preparation-dispatcher",
            daemon=True,
        )
        _WORKER_THREAD.start()
