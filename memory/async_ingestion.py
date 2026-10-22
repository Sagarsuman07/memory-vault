"""Fast upload acceptance for Memory Vault Phase 6.

This module validates and persists the original file + memory + preparation job.
Expensive extraction, chunking, embedding, and indexing are delegated to the
persistent background worker in ``memory.background_worker``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

from config.settings import settings
from database.models import Memory
from database.repositories import (
    create_memory,
    create_preparation_job,
    create_ingestion_source,
    find_duplicate_content,
)
from ingestion.discovery import discover_upload, get_memory_type

MAX_FILE_SIZE = 10 * 1024 * 1024
UPLOAD_DIR = Path(settings.UPLOAD_DIR)


def enqueue_new_memory(uploaded_file, title: str, user_id: str) -> dict[str, str]:
    """Persist an upload and queue preparation without running model inference.

    Returns ``memory_id`` and ``job_id`` immediately after durable persistence.
    The caller should show a queued/pending state; it must not report success
    until the preparation job reaches ``succeeded``.
    """
    if uploaded_file is None:
        raise ValueError("No file was provided.")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("User ID cannot be empty.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Memory title cannot be empty.")

    file_name = getattr(uploaded_file, "name", "")
    getvalue = getattr(uploaded_file, "getvalue", None)
    if not callable(getvalue):
        raise ValueError("Uploaded file does not provide readable bytes.")
    data = getvalue()
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("Uploaded file content must be bytes.")
    data = bytes(data)

    memory_id = str(uuid.uuid4())
    file_path: Path | None = None
    source_created = False
    memory_created = False
    job_created = False
    try:
        descriptor = discover_upload(
            file_name,
            data,
            upload_dir=UPLOAD_DIR,
            source_id=memory_id,
            max_file_size=MAX_FILE_SIZE,
        )
        file_path = Path(descriptor.file_path)
        duplicate_id = find_duplicate_content(user_id, descriptor.sha256)
        warnings = []
        if duplicate_id:
            warnings.append(
                "Identical content already exists in another memory; this upload remains a separate memory."
            )

        # Persist the source first; no extracted text or model calls happen here.
        create_ingestion_source(
            source_id=memory_id,
            user_id=user_id,
            original_file_name=descriptor.file_name,
            extension=descriptor.extension,
            media_type=descriptor.media_type,
            size_bytes=descriptor.size_bytes,
            sha256=descriptor.sha256,
            status="validated",
            stored_file_path=str(file_path),
            duplicate_of_memory_id=duplicate_id,
            warnings_json=json.dumps(warnings, ensure_ascii=False),
        )
        source_created = True
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)

        now = datetime.now(timezone.utc).isoformat()
        create_memory(Memory(
            id=memory_id,
            user_id=user_id,
            memory_type=get_memory_type(descriptor.file_name),
            title=title.strip(),
            file_name=descriptor.file_name,
            file_path=str(file_path),
            summary=None,
            extracted_text=None,
            created_at=now,
            updated_at=now,
        ))
        memory_created = True
        job = create_preparation_job(memory_id, user_id, total_files=1)
        job_created = True

        # Kick the daemon worker after all durable records exist. If startup fails,
        # the queued job remains recoverable on the next application startup.
        try:
            from memory.background_worker import start_background_worker
            start_background_worker()
        except Exception:
            pass

        return {"memory_id": memory_id, "job_id": job.id, "status": job.status.value}
    except Exception:
        # Remove only records/files created by this failed enqueue attempt. Do not
        # touch pre-existing memories or vector records.
        from database.database import get_connection
        if job_created:
            connection = get_connection()
            try:
                connection.execute("DELETE FROM preparation_jobs WHERE memory_id = ? AND user_id = ?", (memory_id, user_id))
                connection.commit()
            finally:
                connection.close()
        if memory_created:
            connection = get_connection()
            try:
                connection.execute("DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, user_id))
                connection.commit()
            finally:
                connection.close()
        if source_created:
            connection = get_connection()
            try:
                connection.execute("DELETE FROM ingestion_sources WHERE source_id = ? AND user_id = ?", (memory_id, user_id))
                connection.commit()
            finally:
                connection.close()
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except OSError:
                pass
        raise
