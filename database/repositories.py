from database.database import get_connection
from database.models import Memory


# ============================================================
# CREATE
# ============================================================

def create_memory(
    memory: Memory
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        INSERT INTO memories (
            id,
            user_id,
            memory_type,
            title,
            file_name,
            file_path,
            summary,
            extracted_text,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory.id,
            memory.user_id,
            memory.memory_type,
            memory.title,
            memory.file_name,
            memory.file_path,
            memory.summary,
            memory.extracted_text,
            memory.created_at,
            memory.updated_at,
        ),
    )


    connection.commit()

    connection.close()


# ============================================================
# GET ONE MEMORY
# ============================================================

def get_memory(
    memory_id: str,
    user_id: str
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM memories
        WHERE id = ?
        AND user_id = ?
        """,
        (
            memory_id,
            user_id,
        ),
    )


    memory = cursor.fetchone()

    connection.close()

    return memory


# ============================================================
# LIST MEMORIES
# ============================================================

def get_all_memories(
    user_id: str
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM memories
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    )


    memories = cursor.fetchall()

    connection.close()

    return memories


# ============================================================
# UPDATE MEMORY
# ============================================================

def update_memory(
    memory_id: str,
    user_id: str,
    title: str,
    summary: str
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        UPDATE memories
        SET title = ?,
            summary = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        AND user_id = ?
        """,
        (
            title,
            summary,
            memory_id,
            user_id,
        ),
    )


    connection.commit()

    rows_updated = cursor.rowcount

    connection.close()

    return rows_updated > 0


# ============================================================
# DELETE MEMORY
# ============================================================

def delete_memory(
    memory_id: str,
    user_id: str
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM memories
        WHERE id = ?
        AND user_id = ?
        """,
        (
            memory_id,
            user_id,
        ),
    )


    connection.commit()

    rows_deleted = cursor.rowcount

    connection.close()

    return rows_deleted > 0


# ============================================================
# PREPARATION JOBS (Phase 1 - additive persistence contract)
# ============================================================

from uuid import uuid4

from database.models import PreparationJob
from memory.memory_models import (
    JobStatus,
    PipelineStage,
    PreparationErrorCategory,
    STAGE_ORDER,
    InvalidStateTransition,
    validate_job_transition,
)


def _job_from_row(row):
    if row is None:
        return None
    return PreparationJob(
        id=row["id"],
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        status=JobStatus(row["status"]),
        current_stage=PipelineStage(row["current_stage"]),
        attempt_count=row["attempt_count"],
        completed_files=row["completed_files"],
        total_files=row["total_files"],
        error_category=(
            PreparationErrorCategory(row["error_category"])
            if row["error_category"] else None
        ),
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def create_preparation_job(
    memory_id: str,
    user_id: str,
    *,
    job_id: str | None = None,
    total_files: int = 1,
) -> PreparationJob:
    """Create a queued job without changing the associated memory or vectors.

    A caller should create the memory record first. This function deliberately
    does not insert/update memories or touch Chroma, uploads, or checkpoints.
    """
    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id must not be empty")
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")
    if total_files < 0:
        raise ValueError("total_files must be zero or greater")

    job_id = job_id or str(uuid4())
    connection = get_connection()
    try:
        memory_row = connection.execute(
            "SELECT id FROM memories WHERE id = ? AND user_id = ?",
            (memory_id, user_id),
        ).fetchone()
        if memory_row is None:
            raise LookupError("Memory not found for the specified user")
        connection.execute(
            """
            INSERT INTO preparation_jobs (
                id, memory_id, user_id, status, current_stage,
                attempt_count, completed_files, total_files
            ) VALUES (?, ?, ?, ?, ?, 0, 0, ?)
            """,
            (
                job_id, memory_id, user_id,
                JobStatus.QUEUED.value,
                PipelineStage.DISCOVERY.value,
                total_files,
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM preparation_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()
        return _job_from_row(row)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_preparation_job(job_id: str, user_id: str) -> PreparationJob | None:
    """Fetch a job scoped to its owner."""
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT * FROM preparation_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()
        return _job_from_row(row)
    finally:
        connection.close()


def list_preparation_jobs(
    user_id: str,
    *,
    statuses: tuple[JobStatus, ...] | None = None,
) -> list[PreparationJob]:
    """List a user's jobs, optionally filtering by one or more statuses."""
    connection = get_connection()
    try:
        if statuses:
            values = tuple(JobStatus(status).value for status in statuses)
            placeholders = ", ".join("?" for _ in values)
            rows = connection.execute(
                f"SELECT * FROM preparation_jobs WHERE user_id = ? "
                f"AND status IN ({placeholders}) ORDER BY created_at DESC, id",
                (user_id, *values),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM preparation_jobs WHERE user_id = ? "
                "ORDER BY created_at DESC, id",
                (user_id,),
            ).fetchall()
        return [_job_from_row(row) for row in rows]
    finally:
        connection.close()


def transition_preparation_job(
    job_id: str,
    user_id: str,
    target_status: JobStatus | str,
    *,
    error_category: PreparationErrorCategory | str | None = None,
    error_message: str | None = None,
) -> PreparationJob:
    """Apply a validated lifecycle transition atomically.

    Pass only a sanitized diagnostic message; never pass credentials, raw file
    content, or environment-variable values as error_message.
    """
    target = JobStatus(target_status)
    category = (
        PreparationErrorCategory(error_category) if error_category else None
    )
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM preparation_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()
        if row is None:
            raise LookupError("Preparation job not found")

        current = JobStatus(row["status"])
        validate_job_transition(current, target)
        if target == JobStatus.FAILED and category is None:
            raise ValueError("Failed jobs require an error_category")
        if target == JobStatus.SUCCEEDED:
            if PipelineStage(row["current_stage"]) != PipelineStage.INDEXING:
                raise ValueError("A job can succeed only after indexing has been reached")
            if row["completed_files"] != row["total_files"]:
                raise ValueError("A job cannot succeed until all files are complete")

        if target == JobStatus.RUNNING:
            # A new attempt starts from the first stage; retry progress is not
            # represented as progress on the failed attempt.
            connection.execute(
                """UPDATE preparation_jobs SET status = ?, current_stage = ?,
                   attempt_count = attempt_count + 1, completed_files = 0,
                   error_category = NULL, error_message = NULL,
                   started_at = CURRENT_TIMESTAMP, finished_at = NULL,
                   updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND user_id = ?""",
                (target.value, PipelineStage.DISCOVERY.value, job_id, user_id),
            )
        elif target == JobStatus.RETRYING:
            connection.execute(
                """UPDATE preparation_jobs SET status = ?, error_category = ?,
                   error_message = ?, finished_at = NULL,
                   updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND user_id = ?""",
                (target.value, category.value if category else None,
                 error_message, job_id, user_id),
            )
        elif target == JobStatus.FAILED:
            connection.execute(
                """UPDATE preparation_jobs SET status = ?, error_category = ?,
                   error_message = ?, finished_at = CURRENT_TIMESTAMP,
                   updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND user_id = ?""",
                (target.value, category.value, error_message, job_id, user_id),
            )
        elif target == JobStatus.SUCCEEDED:
            connection.execute(
                """UPDATE preparation_jobs SET status = ?, current_stage = ?,
                   error_category = NULL, error_message = NULL,
                   finished_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND user_id = ?""",
                (target.value, PipelineStage.COMPLETE.value, job_id, user_id),
            )
        else:
            raise InvalidStateTransition(f"Unsupported target state: {target.value}")

        updated = connection.execute(
            "SELECT * FROM preparation_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()
        connection.commit()
        return _job_from_row(updated)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def update_preparation_job_progress(
    job_id: str,
    user_id: str,
    stage: PipelineStage | str,
    *,
    completed_files: int | None = None,
    total_files: int | None = None,
) -> PreparationJob:
    """Persist stage and file-count progress for a running job."""
    target_stage = PipelineStage(stage)
    if target_stage == PipelineStage.COMPLETE:
        raise ValueError("Use transition_preparation_job(..., 'succeeded') to complete a job")
    if completed_files is not None and completed_files < 0:
        raise ValueError("completed_files must be zero or greater")
    if total_files is not None and total_files < 0:
        raise ValueError("total_files must be zero or greater")

    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM preparation_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()
        if row is None:
            raise LookupError("Preparation job not found")
        if JobStatus(row["status"]) != JobStatus.RUNNING:
            raise InvalidStateTransition("Progress can only be updated for running jobs")

        current_stage = PipelineStage(row["current_stage"])
        current_index = STAGE_ORDER.index(current_stage)
        target_index = STAGE_ORDER.index(target_stage)
        if target_index < current_index or target_index > current_index + 1:
            raise InvalidStateTransition(
                f"Invalid stage transition: {current_stage.value} -> {target_stage.value}"
            )

        new_completed = row["completed_files"] if completed_files is None else completed_files
        new_total = row["total_files"] if total_files is None else total_files
        if new_completed > new_total:
            raise ValueError("completed_files cannot exceed total_files")
        if new_completed < row["completed_files"]:
            raise ValueError("completed_files cannot decrease during an attempt")
        if new_total < row["total_files"] and new_completed > new_total:
            raise ValueError("total_files cannot be lower than completed_files")

        connection.execute(
            """UPDATE preparation_jobs SET current_stage = ?, completed_files = ?,
               total_files = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ? AND user_id = ?""",
            (target_stage.value, new_completed, new_total, job_id, user_id),
        )
        updated = connection.execute(
            "SELECT * FROM preparation_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()
        connection.commit()
        return _job_from_row(updated)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

# ============================================================
# INGESTION SOURCES (Phase 2)
# ============================================================

from database.models import IngestionSource

_VALID_SOURCE_STATUSES = {"discovered", "validated", "loaded", "skipped", "failed"}
_SOURCE_UPDATE_FIELDS = {
    "memory_id", "stored_file_path", "duplicate_of_memory_id", "status",
    "warnings_json", "metadata_json", "error_category", "error_message",
}


def _ingestion_source_from_row(row):
    if row is None:
        return None
    return IngestionSource(
        source_id=row["source_id"],
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        original_file_name=row["original_file_name"],
        extension=row["extension"],
        media_type=row["media_type"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        status=row["status"],
        stored_file_path=row["stored_file_path"],
        duplicate_of_memory_id=row["duplicate_of_memory_id"],
        warnings_json=row["warnings_json"],
        metadata_json=row["metadata_json"],
        error_category=row["error_category"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def create_ingestion_source(
    *,
    source_id: str,
    user_id: str,
    original_file_name: str,
    extension: str,
    media_type: str,
    size_bytes: int,
    sha256: str,
    status: str = "validated",
    memory_id: str | None = None,
    stored_file_path: str | None = None,
    duplicate_of_memory_id: str | None = None,
    warnings_json: str = "[]",
    metadata_json: str = "{}",
    error_category: str | None = None,
    error_message: str | None = None,
) -> IngestionSource:
    """Persist source metadata without storing file content or credentials."""
    for name, value in (("source_id", source_id), ("user_id", user_id),
                        ("original_file_name", original_file_name), ("extension", extension),
                        ("media_type", media_type), ("sha256", sha256)):
        if not value or not value.strip():
            raise ValueError(f"{name} must not be empty")
    if size_bytes < 0:
        raise ValueError("size_bytes must be zero or greater")
    if status not in _VALID_SOURCE_STATUSES:
        raise ValueError(f"Unsupported ingestion source status: {status}")

    connection = get_connection()
    try:
        connection.execute(
            """INSERT INTO ingestion_sources (
                source_id, memory_id, user_id, original_file_name, extension,
                media_type, size_bytes, sha256, status, stored_file_path,
                duplicate_of_memory_id, warnings_json, metadata_json, error_category, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_id, memory_id, user_id, original_file_name, extension,
             media_type, size_bytes, sha256, status, stored_file_path,
             duplicate_of_memory_id, warnings_json, metadata_json, error_category, error_message),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM ingestion_sources WHERE source_id = ? AND user_id = ?",
            (source_id, user_id),
        ).fetchone()
        return _ingestion_source_from_row(row)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_ingestion_source(source_id: str, user_id: str) -> IngestionSource | None:
    """Read source tracking data only within the requesting user's scope."""
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT * FROM ingestion_sources WHERE source_id = ? AND user_id = ?",
            (source_id, user_id),
        ).fetchone()
        return _ingestion_source_from_row(row)
    finally:
        connection.close()


def get_ingestion_source_for_memory(memory_id: str, user_id: str) -> IngestionSource | None:
    connection = get_connection()
    try:
        row = connection.execute(
            """SELECT * FROM ingestion_sources
               WHERE memory_id = ? AND user_id = ?
               ORDER BY updated_at DESC LIMIT 1""",
            (memory_id, user_id),
        ).fetchone()
        return _ingestion_source_from_row(row)
    finally:
        connection.close()


def find_duplicate_content(
    user_id: str,
    sha256: str,
    *,
    exclude_source_id: str | None = None,
) -> str | None:
    """Return an existing same-user memory ID with identical content, if any.

    This is informational only: callers must not merge or reject a new memory
    solely because its content hash matches an existing memory.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty")
    if not sha256 or not sha256.strip():
        raise ValueError("sha256 must not be empty")
    connection = get_connection()
    try:
        query = """SELECT memory_id FROM ingestion_sources
                   WHERE user_id = ? AND sha256 = ? AND status = 'loaded'
                     AND memory_id IS NOT NULL"""
        params: list[str] = [user_id, sha256]
        if exclude_source_id:
            query += " AND source_id != ?"
            params.append(exclude_source_id)
        query += " ORDER BY created_at ASC LIMIT 1"
        row = connection.execute(query, tuple(params)).fetchone()
        return row["memory_id"] if row else None
    finally:
        connection.close()


def update_ingestion_source(source_id: str, user_id: str, **fields) -> bool:
    """Update an owned source's status/association using an allowlist of fields."""
    if not source_id or not source_id.strip() or not user_id or not user_id.strip():
        raise ValueError("source_id and user_id must not be empty")
    unknown = set(fields) - _SOURCE_UPDATE_FIELDS
    if unknown:
        raise ValueError(f"Unsupported ingestion source fields: {', '.join(sorted(unknown))}")
    if not fields:
        return False
    if "status" in fields and fields["status"] not in _VALID_SOURCE_STATUSES:
        raise ValueError(f"Unsupported ingestion source status: {fields['status']}")
    assignments = [f"{name} = ?" for name in fields]
    values = list(fields.values())
    assignments.append("updated_at = CURRENT_TIMESTAMP")
    connection = get_connection()
    try:
        cursor = connection.execute(
            f"UPDATE ingestion_sources SET {', '.join(assignments)} "
            "WHERE source_id = ? AND user_id = ?",
            (*values, source_id, user_id),
        )
        connection.commit()
        return cursor.rowcount > 0
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
