"""Fast upload acceptance for Memory Vault using private Cloudflare R2."""
from __future__ import annotations
from datetime import datetime, timezone
from io import BytesIO
import json, logging, time, uuid
from config.logging_config import get_logger, log_event
from config.settings import settings
from database.models import Memory
from database.repositories import create_memory, create_preparation_job, create_ingestion_source, find_duplicate_content
from ingestion.discovery import discover_upload, get_memory_type
from storage.file_store import build_memory_object_key, delete_file, upload_file

logger = get_logger(__name__)
MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE_BYTES


def enqueue_new_memory(uploaded_file, title: str, user_id: str) -> dict[str, str]:
    """Upload the original file to R2, persist its key, and queue preparation."""
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
    if not isinstance(data, bytes):
        data = bytes(data)

    started = time.perf_counter()
    memory_id = str(uuid.uuid4())
    original_key = None
    source_created = memory_created = job_created = False
    try:
        descriptor = discover_upload(file_name, data, upload_dir=None,
                                     source_id=memory_id, max_file_size=MAX_FILE_SIZE)
        original_key = build_memory_object_key(
            user_id, memory_id, "original", f"source{descriptor.extension}"
        )
        duplicate_id = find_duplicate_content(user_id, descriptor.sha256)
        warnings = []
        if duplicate_id:
            warnings.append("Identical content already exists in another memory; this upload remains a separate memory.")

        stream = uploaded_file
        if not callable(getattr(stream, "read", None)) or not callable(getattr(stream, "seek", None)):
            stream = BytesIO(data)
        stream.seek(0)
        upload_file(stream, original_key, descriptor.media_type)

        create_ingestion_source(
            source_id=memory_id, user_id=user_id,
            original_file_name=descriptor.file_name, extension=descriptor.extension,
            media_type=descriptor.media_type, size_bytes=descriptor.size_bytes,
            sha256=descriptor.sha256, status="validated", stored_file_path=original_key,
            duplicate_of_memory_id=duplicate_id,
            warnings_json=json.dumps(warnings, ensure_ascii=False),
        )
        source_created = True
        now = datetime.now(timezone.utc).isoformat()
        create_memory(Memory(
            id=memory_id, user_id=user_id, memory_type=get_memory_type(descriptor.file_name),
            title=title.strip(), file_name=descriptor.file_name, file_path=original_key,
            summary=None, extracted_text=None, created_at=now, updated_at=now,
            original_file_key=original_key,
        ))
        memory_created = True
        job = create_preparation_job(memory_id, user_id, total_files=1)
        job_created = True
        log_event(logger, "INGEST_UPLOAD_QUEUED", job=job.id, memory=memory_id,
                  user=user_id, file=descriptor.file_name, bytes=descriptor.size_bytes,
                  seconds=time.perf_counter() - started)
        try:
            from memory.background_worker import start_background_worker
            start_background_worker()
        except Exception:
            pass
        return {"memory_id": memory_id, "job_id": job.id, "status": job.status.value}
    except Exception as error:
        log_event(logger, "INGEST_UPLOAD_ERROR", level=logging.ERROR, memory=memory_id,
                  user=user_id, file=file_name or "unknown", error=type(error).__name__)
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
        try:
            delete_file(original_key)
        except Exception:
            logger.warning("Failed to clean up a partial R2 upload for memory %s", memory_id)
        raise
