from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
import io
import json
import uuid
import wave
import tempfile

from langchain_groq import ChatGroq

from config.logging_config import get_logger, log_event
from config.settings import settings

from database.database import get_connection
from database.models import Memory
from memory.memory_models import SourceDocument

from database.repositories import (
    create_memory,
    get_memory,
    get_all_memories,
    update_memory as update_memory_record,
    delete_memory as delete_memory_record,
    create_ingestion_source,
    find_duplicate_content,
    get_ingestion_source_for_memory,
    update_ingestion_source,
)

from ingestion.pipeline import (
    SourceValidationError,
    detect_source_change,
    discover_upload,
    extract_content,
    process_content,
    get_memory_type,
)


from prompts.summary import SUMMARY_PROMPT
from prompts.safety import SAFETY_PROMPT

from rag.chunker import split_text

from rag.vector_store import (
    index_memory,
    delete_memory_index,
)

from storage.file_store import build_memory_object_key, delete_file as delete_storage_file, download_to_temp, download_text, upload_file, upload_text




logger = get_logger(__name__)



ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".mp3",
    ".wav",
    ".m4a",
    ".mpeg",
    ".mpga",
    ".webm",
    ".ogg",
    ".flac",
}


MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE_BYTES


# ============================================================
# Create Memory — Phase 2 source discovery and tracking
# ============================================================

def create_new_memory(
    uploaded_file, title: str, user_id: str,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """Synchronous compatibility path using R2 and temporary processor files."""
    if uploaded_file is None:
        raise ValueError("No file was provided.")
    if not user_id or not user_id.strip():
        raise ValueError("User ID cannot be empty.")
    if not title or not title.strip():
        raise ValueError("Memory title cannot be empty.")
    file_name = getattr(uploaded_file, "name", "")
    file_bytes = uploaded_file.getvalue()
    if not isinstance(file_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("Uploaded file content must be bytes.")
    if not isinstance(file_bytes, bytes):
        file_bytes = bytes(file_bytes)

    memory_id = str(uuid.uuid4())
    original_key = extracted_key = None
    source_created = False
    stage = "validation"
    try:
        descriptor = discover_upload(file_name, file_bytes, upload_dir=None,
                                     source_id=memory_id, max_file_size=MAX_FILE_SIZE)
        memory_type = get_memory_type(descriptor.file_name)
        original_key = build_memory_object_key(user_id, memory_id, "original",
                                               f"source{descriptor.extension}")
        stream = uploaded_file
        if not callable(getattr(stream, "read", None)) or not callable(getattr(stream, "seek", None)):
            stream = io.BytesIO(file_bytes)
        stream.seek(0)
        upload_file(stream, original_key, descriptor.media_type)
        duplicate_id = find_duplicate_content(user_id, descriptor.sha256)
        warnings = []
        if duplicate_id:
            warnings.append("Identical content already exists in another memory; this upload was retained as a separate memory.")
        create_ingestion_source(
            source_id=descriptor.source_id, user_id=user_id,
            original_file_name=descriptor.file_name, extension=descriptor.extension,
            media_type=descriptor.media_type, size_bytes=descriptor.size_bytes,
            sha256=descriptor.sha256, status="validated", stored_file_path=original_key,
            duplicate_of_memory_id=duplicate_id, warnings_json=json.dumps(warnings, ensure_ascii=False),
        )
        source_created = True
        stage = "extraction"
        if progress_callback:
            progress_callback("extracting")
        with download_to_temp(original_key, suffix=descriptor.extension) as source_path:
            processed = process_content(source_path)
        source_document = SourceDocument(
            source_id=descriptor.source_id, memory_id=memory_id,
            file_name=descriptor.file_name, media_type=descriptor.media_type,
            text=processed.text, metadata=dict(processed.metadata),
            extraction_status=processed.status, warnings=tuple(processed.warnings),
        )
        extracted_text = source_document.text
        if not extracted_text.strip():
            raise ValueError("No usable text was extracted.")
        update_ingestion_source(
            descriptor.source_id, user_id,
            metadata_json=json.dumps(dict(source_document.metadata), ensure_ascii=False),
            warnings_json=json.dumps(list(dict.fromkeys(warnings + list(source_document.warnings))), ensure_ascii=False),
        )
        extracted_key = build_memory_object_key(user_id, memory_id, "extracted", "content.txt")
        upload_text(extracted_text, extracted_key)
        now = datetime.now(timezone.utc).isoformat()
        memory = Memory(
            id=memory_id, user_id=user_id, memory_type=memory_type, title=title.strip(),
            file_name=descriptor.file_name, file_path=original_key, summary=None,
            extracted_text=None, created_at=now, updated_at=now,
            original_file_key=original_key, extracted_text_key=extracted_key,
        )
        create_memory(memory)
        stage = "chunking"
        chunks = split_text(extracted_text, metadata=source_document.metadata)
        if not chunks:
            raise ValueError("No chunks could be created for indexing.")
        stage = "indexing"
        index_memory(memory_id=memory.id, user_id=memory.user_id,
                     memory_type=memory.memory_type, chunks=chunks,
                     progress_callback=progress_callback)
        update_ingestion_source(
            descriptor.source_id, user_id, memory_id=memory_id, status="loaded",
            stored_file_path=original_key, error_category=None, error_message=None,
        )
        if progress_callback:
            progress_callback("done")
        return memory
    except Exception as error:
        if not source_created:
            try:
                safe_name = Path(str(file_name).replace(chr(92), "/")).name or "unknown"
                extension = Path(safe_name).suffix.lower() or "(none)"
                from hashlib import sha256
                from ingestion.discovery import get_media_type
                try:
                    media_type = get_media_type(safe_name)
                except Exception:
                    media_type = "application/octet-stream"
                create_ingestion_source(
                    source_id=memory_id, user_id=user_id, original_file_name=safe_name,
                    extension=extension, media_type=media_type, size_bytes=len(file_bytes),
                    sha256=sha256(file_bytes).hexdigest(), status="failed",
                    error_category="validation" if isinstance(error, (ValueError, SourceValidationError)) else "internal",
                    error_message=str(error) if isinstance(error, (ValueError, SourceValidationError)) else "Source discovery failed unexpectedly.",
                )
                source_created = True
            except Exception:
                pass
        try:
            delete_memory_index(memory_id)
        except Exception:
            pass
        for key in (extracted_key, original_key):
            try:
                delete_storage_file(key)
            except Exception:
                pass
        try:
            delete_memory_record(memory_id, user_id)
        except Exception:
            pass
        if source_created:
            category = {"extraction": "extraction", "chunking": "chunking", "indexing": "indexing"}.get(
                stage, "validation" if isinstance(error, (ValueError, SourceValidationError)) else "internal"
            )
            safe_message = str(error) if category == "validation" else (
                "Source could not be extracted." if category == "extraction" else
                "Loaded source could not be prepared for indexing." if category == "chunking" else
                "Memory indexing failed." if category == "indexing" else
                "An unexpected ingestion error occurred."
            )
            try:
                update_ingestion_source(memory_id, user_id, memory_id=None,
                                        stored_file_path=None, status="failed",
                                        error_category=category, error_message=safe_message)
            except Exception:
                pass
        raise


def check_memory_source_change(
    memory_id: str,
    user_id: str,
    file_path: Optional[str] = None,
):
    """Compare a tracked memory source's saved hash with current file contents.

    This reports whether a source changed; it deliberately does not reprocess or
    re-index it automatically. That update workflow belongs to later phases.
    """
    if not memory_id or not memory_id.strip():
        raise ValueError("Memory ID cannot be empty.")
    if not user_id or not user_id.strip():
        raise ValueError("User ID cannot be empty.")
    source = get_ingestion_source_for_memory(memory_id, user_id)
    if source is None:
        raise ValueError("No ingestion source is tracked for this memory.")
    if file_path:
        return detect_source_change(source.sha256, file_path, source.source_id)
    object_key = source.stored_file_path
    if not object_key:
        raise ValueError("The source object key is unavailable.")
    with download_to_temp(object_key, suffix=Path(source.original_file_name).suffix) as current_path:
        return detect_source_change(source.sha256, current_path, source.source_id)


# ============================================================
# Get Memory
# ============================================================

def get_memory_by_id(
    memory_id: str,
    user_id: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_id or not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    memory = get_memory(
        memory_id,
        user_id,
    )

    if memory is None:
        raise ValueError(
            "Memory not found."
        )

    return memory


def get_memory_extracted_text(memory_id: str, user_id: str) -> str | None:
    """Retrieve extracted content from R2 only when explicitly needed."""
    memory = get_memory(memory_id, user_id)
    if memory is None:
        raise ValueError("Memory not found.")
    key = memory.get("extracted_text_key")
    return download_text(key) if key else memory.get("extracted_text")


# ============================================================
# List Memories
# ============================================================

def list_memories(
    user_id: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    memories = get_all_memories(
        user_id
    )

    # Always keep dashboard chronology deterministic.
    return sorted(
        memories,
        key=lambda memory: memory["created_at"],
        reverse=True,
    )


# ============================================================
# Update Memory
# ============================================================

def update_memory(
    memory_id: str,
    user_id: str,
    title: str,
    summary: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_id or not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    if not title or not title.strip():
        raise ValueError(
            "Memory title cannot be empty."
        )

    updated = update_memory_record(
        memory_id=memory_id,
        user_id=user_id,
        title=title.strip(),
        summary=summary.strip(),
    )

    if not updated:
        raise ValueError(
            "Memory not found."
        )

    return get_memory(
        memory_id,
        user_id,
    )


# ============================================================
# Cleanup of ingestion records left behind by a deleted memory
# ============================================================

def _cleanup_deleted_memory_records(
    memory_id: str,
    user_id: str,
):
    """
    Remove the ingestion bookkeeping rows that belong to ONE deleted memory.

    * ingestion_sources: the upload record (source_id equals the memory id) and
      any record linked to the memory. Without this, a stale 'loaded' record
      would keep advertising a duplicate that no longer exists.
    * preparation_jobs: the job rows of that memory.

    Everything is scoped by user_id. This runs only AFTER the memory itself has
    been deleted, and it is best-effort: a failure is logged and never turns a
    successful delete into an error. Audit rows of FAILED uploads (which have
    no memory) are intentionally not touched.
    """

    sources_deleted = 0
    jobs_deleted = 0

    try:
        connection = get_connection()

        try:
            cursor = connection.execute(
                "DELETE FROM ingestion_sources "
                "WHERE user_id = ? AND (source_id = ? OR memory_id = ?) "
                "AND status != 'failed'",
                (user_id, memory_id, memory_id),
            )
            sources_deleted = max(0, cursor.rowcount)

            cursor = connection.execute(
                "DELETE FROM preparation_jobs "
                "WHERE user_id = ? AND memory_id = ?",
                (user_id, memory_id),
            )
            jobs_deleted = max(0, cursor.rowcount)

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        log_event(
            logger,
            "MEMORY_DELETE_CLEANUP",
            memory=memory_id,
            user=user_id,
            ingestion_sources_removed=sources_deleted,
            preparation_jobs_removed=jobs_deleted,
        )

    except Exception as error:
        # The memory is already gone; never raise from cleanup.
        log_event(
            logger,
            "MEMORY_DELETE_CLEANUP",
            30,
            memory=memory_id,
            user=user_id,
            status="failed",
            error=type(error).__name__,
        )


# ============================================================
# Delete Single Memory
# ============================================================

def delete_memory(
    memory_id: str,
    user_id: str,
):
    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_id or not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    # User-scoped ownership validation.
    memory = get_memory(
        memory_id,
        user_id,
    )

    if memory is None:
        raise ValueError(
            "Memory not found."
        )

    # ========================================================
    # Delete vector records
    # ========================================================

    delete_memory_index(memory_id)

    # ========================================================
    # Delete original file
    # ========================================================

    try:
        delete_storage_file(memory.get("original_file_key") or memory.get("file_path"))
        delete_storage_file(memory.get("extracted_text_key"))
    except Exception as error:
        raise RuntimeError("Memory files could not be deleted from object storage.") from error

    # ========================================================
    # Delete database record
    # ========================================================

    deleted = delete_memory_record(
        memory_id,
        user_id,
    )

    if not deleted:
        raise ValueError(
            "Memory could not be deleted."
        )

    # Remove the ingestion records that belonged to this memory.
    _cleanup_deleted_memory_records(
        memory_id,
        user_id,
    )


# ============================================================
# Bulk Delete
# ============================================================

def delete_memories(
    memory_ids: list[str],
    user_id: str,
):
    """
    Delete multiple memories.

    Every memory is resolved using the current user_id
    before deletion. This prevents one user from deleting
    another user's memory.
    """

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if not memory_ids:
        return 0

    unique_memory_ids = list(
        dict.fromkeys(
            memory_ids
        )
    )

    # ========================================================
    # Validate ownership of ALL memories first.
    # ========================================================

    memories = []

    for memory_id in unique_memory_ids:

        if (
            not memory_id
            or not memory_id.strip()
        ):
            raise ValueError(
                "Memory ID cannot be empty."
            )

        memory = get_memory(
            memory_id,
            user_id,
        )

        if memory is None:
            raise ValueError(
                "One or more selected memories "
                "do not belong to the current user."
            )

        memories.append(
            memory
        )

    # ========================================================
    # Delete only after ownership validation succeeds.
    # ========================================================

    for memory in memories:

        delete_memory(
            memory_id=memory["id"],
            user_id=user_id,
        )

    return len(
        memories
    )


# ============================================================
# One-off cleanup of records left by memories deleted EARLIER
# ============================================================

def purge_orphaned_ingestion_records(
    user_id: Optional[str] = None,
    dry_run: bool = True,
):
    """
    Remove ingestion records whose memory no longer exists.

    Targets only:
      * ingestion_sources with status 'loaded' whose memory is gone
        (these are the stale "duplicate" records), and
      * finished (succeeded/failed) preparation_jobs whose memory is gone.

    Never touches memories, vectors, files, uploads, active jobs, or the audit
    rows of failed uploads. Scope it with ``user_id`` if desired.

    ``dry_run=True`` (the default) only COUNTS what would be removed. Call with
    ``dry_run=False`` to delete. Returns a dict of counts.

    Example:
        from memory.memory_service import purge_orphaned_ingestion_records
        print(purge_orphaned_ingestion_records())                # preview
        print(purge_orphaned_ingestion_records(dry_run=False))   # delete
    """

    user_filter = ""
    params: tuple = ()

    if user_id is not None:
        if not user_id.strip():
            raise ValueError(
                "User ID cannot be empty."
            )
        user_filter = " AND user_id = ?"
        params = (user_id,)

    sources_where = (
        "status = 'loaded' AND memory_id IS NOT NULL"
        " AND memory_id NOT IN (SELECT id FROM memories)"
        " AND source_id NOT IN (SELECT id FROM memories)"
        + user_filter
    )
    jobs_where = (
        "status IN ('succeeded', 'failed')"
        " AND memory_id NOT IN (SELECT id FROM memories)"
        + user_filter
    )

    connection = get_connection()

    try:
        sources = connection.execute(
            f"SELECT COUNT(*) AS n FROM ingestion_sources WHERE {sources_where}",
            params,
        ).fetchone()["n"]

        jobs = connection.execute(
            f"SELECT COUNT(*) AS n FROM preparation_jobs WHERE {jobs_where}",
            params,
        ).fetchone()["n"]

        if not dry_run and (sources or jobs):
            try:
                connection.execute(
                    f"DELETE FROM ingestion_sources WHERE {sources_where}",
                    params,
                )
                connection.execute(
                    f"DELETE FROM preparation_jobs WHERE {jobs_where}",
                    params,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    finally:
        connection.close()

    log_event(
        logger,
        "MEMORY_ORPHAN_PURGE",
        dry_run=dry_run,
        user=user_id,
        ingestion_sources=sources,
        preparation_jobs=jobs,
    )

    return {
        "dry_run": dry_run,
        "ingestion_sources": sources,
        "preparation_jobs": jobs,
    }


# ============================================================
# Demo Helpers
# ============================================================

def _create_demo_text_file(file_name: str, content: str, directory: Path) -> Path:
    path = directory / file_name
    path.write_text(content, encoding="utf-8")
    return path


def _create_demo_pdf(file_name: str, directory: Path) -> Path:
    """Create a small real PDF for the demo in a temporary directory."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = directory / file_name
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 16)
    pdf.drawString(60, 800, "Delhi Travel Plan")
    pdf.setFont("Helvetica", 11)
    lines = [
        "Travel destination: Delhi",
        "Flight: 6E123",
        "Travel date: 15 September 2026",
        "Hotel: Hotel ABC",
        "Hotel stay: 15-18 September 2026",
        "Important place: India Gate",
        "Estimated hotel cost: INR 5,000 per night",
    ]
    y = 770
    for line in lines:
        pdf.drawString(60, y, line)
        y -= 22
    pdf.save()
    return path


def _create_demo_image(
    file_name: str, title: str, lines: list[str], directory: Path
) -> Path:
    """Create a real PNG demo image in a temporary directory."""
    from PIL import Image, ImageDraw

    path = directory / file_name
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 70), title, fill="black")
    y = 180
    for line in lines:
        draw.text((90, y), line, fill="black")
        y += 70
    image.save(path, format="PNG")
    return path


def _create_demo_audio(file_name: str, directory: Path) -> Path:
    """Create a valid, silent WAV container for the deterministic demo."""
    path = directory / file_name
    sample_rate = 16000
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * sample_rate)
    return path


def _create_memory_from_demo_text(
    title: str,
    file_path: Path,
    memory_type: str,
    extracted_text: str,
    user_id: str,
):
    """Create a deterministic demo memory with both objects stored in R2."""
    from ingestion.discovery import discover_upload

    memory_id = str(uuid.uuid4())
    descriptor = discover_upload(
        file_path.name,
        file_path.read_bytes(),
        upload_dir=None,
        source_id=memory_id,
        max_file_size=MAX_FILE_SIZE,
    )
    original_key = build_memory_object_key(
        user_id, memory_id, "original", f"source{descriptor.extension}"
    )
    extracted_key = build_memory_object_key(
        user_id, memory_id, "extracted", "content.txt"
    )
    memory_created = source_created = False

    try:
        with file_path.open("rb") as source_stream:
            upload_file(source_stream, original_key, descriptor.media_type)
        upload_text(extracted_text, extracted_key)

        now = datetime.now(timezone.utc).isoformat()
        memory = Memory(
            id=memory_id,
            user_id=user_id,
            memory_type=memory_type,
            title=title,
            file_name=descriptor.file_name,
            file_path=original_key,
            summary=None,
            extracted_text=None,
            created_at=now,
            updated_at=now,
            original_file_key=original_key,
            extracted_text_key=extracted_key,
        )
        create_memory(memory)
        memory_created = True

        create_ingestion_source(
            source_id=memory_id,
            user_id=user_id,
            original_file_name=descriptor.file_name,
            extension=descriptor.extension,
            media_type=descriptor.media_type,
            size_bytes=descriptor.size_bytes,
            sha256=descriptor.sha256,
            status="validated",
            memory_id=memory_id,
            stored_file_path=original_key,
            metadata_json=json.dumps(
                {"extraction_method": "deterministic_demo"}, ensure_ascii=False
            ),
        )
        source_created = True

        chunks = split_text(extracted_text)
        if not chunks:
            raise ValueError("No chunks could be created.")

        index_memory(
            memory_id=memory.id,
            user_id=memory.user_id,
            memory_type=memory.memory_type,
            chunks=chunks,
        )
        if not update_ingestion_source(
            memory_id,
            user_id,
            status="loaded",
            memory_id=memory_id,
            stored_file_path=original_key,
            error_category=None,
            error_message=None,
        ):
            raise RuntimeError("Demo ingestion source could not be marked as loaded.")

        return memory

    except Exception:
        try:
            delete_memory_index(memory_id)
        except Exception:
            pass
        if memory_created:
            try:
                delete_memory_record(memory_id, user_id)
            except Exception:
                pass
        if source_created:
            try:
                update_ingestion_source(
                    memory_id,
                    user_id,
                    memory_id=None,
                    stored_file_path=None,
                    status="failed",
                    error_category="internal",
                    error_message="Deterministic demo preparation failed.",
                )
            except Exception:
                pass
        for object_key in (extracted_key, original_key):
            try:
                delete_storage_file(object_key)
            except Exception:
                pass
        raise


def load_demo_memories(user_id: str):
    """Create repeatable demo memories without retaining files on local disk."""
    if not user_id or not user_id.strip():
        raise ValueError("User ID cannot be empty.")

    existing_memories = list_memories(user_id)
    existing_demo_titles = {
        memory["title"]
        for memory in existing_memories
        if memory["title"].startswith("[Demo]")
    }

    demo_definitions = [
        {
            "title": "[Demo] Flight Ticket",
            "type": "image",
            "file_name": "demo_flight.png",
            "text": "Flight ticket. Flight number 6E123. Route Hyderabad to Delhi. Travel date 15 September 2026. Ticket price INR 8,500.",
        },
        {
            "title": "[Demo] Hotel Booking",
            "type": "document",
            "file_name": "demo_hotel.txt",
            "text": "Hotel booking. Hotel ABC in Delhi. Stay from 15 September 2026 to 18 September 2026. Room cost INR 5,000 per night.",
        },
        {
            "title": "[Demo] Sony Headphones",
            "type": "image",
            "file_name": "demo_headphones.png",
            "text": "Product information. Sony headphones. Listed price INR 25,000.",
        },
        {
            "title": "[Demo] Delhi Trip Voice Note",
            "type": "audio",
            "file_name": "demo_voice_note.wav",
            "text": "Remember to visit India Gate during the Delhi trip.",
        },
        {
            "title": "[Demo] Delhi Travel Plan",
            "type": "document",
            "file_name": "demo_travel_plan.pdf",
            "text": "Delhi travel plan. Flight 6E123 from Hyderabad to Delhi on 15 September 2026. Hotel ABC from 15 to 18 September. Remember to visit India Gate.",
        },
    ]

    created = []
    with tempfile.TemporaryDirectory(prefix="memory-vault-demo-") as temp_dir:
        demo_directory = Path(temp_dir)

        for demo in demo_definitions:
            if demo["title"] in existing_demo_titles:
                continue

            file_name = demo["file_name"]
            file_path = demo_directory / file_name
            title = demo["title"]

            if title == "[Demo] Flight Ticket":
                _create_demo_image(
                    file_name, "Flight Ticket",
                    ["IndiGo 6E123", "Hyderabad -> Delhi", "15 September 2026", "INR 8,500"],
                    demo_directory,
                )
            elif title == "[Demo] Hotel Booking":
                _create_demo_text_file(file_name, demo["text"], demo_directory)
            elif title == "[Demo] Sony Headphones":
                _create_demo_image(
                    file_name, "Sony Headphones",
                    ["Sony WH Series", "Premium Headphones", "Price: INR 25,000"],
                    demo_directory,
                )
            elif title == "[Demo] Delhi Trip Voice Note":
                _create_demo_audio(file_name, demo_directory)
                memory = _create_memory_from_demo_text(
                    title=title,
                    file_path=file_path,
                    memory_type="audio",
                    extracted_text=demo["text"],
                    user_id=user_id,
                )
                created.append(memory)
                continue
            elif title == "[Demo] Delhi Travel Plan":
                _create_demo_pdf(file_name, demo_directory)

            class DemoFile:
                def __init__(self, path: Path):
                    self.name = path.name
                    self._path = path

                def getvalue(self):
                    return self._path.read_bytes()

            memory = create_new_memory(
                uploaded_file=DemoFile(file_path),
                title=title,
                user_id=user_id,
            )
            created.append(memory)

    return created


# ============================================================
# Summary Model
# ============================================================

def get_summary_model():

    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return ChatGroq(
        model=settings.GROQ_LLM_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )


# ============================================================
# Safety Model
# ============================================================

def get_safety_model():

    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    return ChatGroq(
        model=settings.GROQ_LLM_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )


# ============================================================
# Content Safety
# ============================================================

def check_content_safety(
    content: str,
) -> bool:

    if not content or not content.strip():
        return False

    if not settings.SAFETY_ENABLED:
        return True

    model = get_safety_model()

    prompt = SAFETY_PROMPT.invoke(
        {
            "content": content,
        }
    )

    response = model.invoke(
        prompt
    )

    result = response.content

    if not isinstance(
        result,
        str,
    ):
        return False

    return (
        result.strip().upper()
        == "ALLOW"
    )


# ============================================================
# Generate Summary
# ============================================================

def generate_memory_summary(
    memory_id: str,
    user_id: str,
):

    memory = get_memory_by_id(
        memory_id,
        user_id,
    )

    extracted_text = get_memory_extracted_text(memory_id, user_id)

    if (
        not extracted_text
        or not extracted_text.strip()
    ):
        raise ValueError(
            "No extracted content is available "
            "for summarization."
        )

    if not check_content_safety(
        extracted_text
    ):
        raise ValueError(
            "This memory cannot be summarized "
            "because its content did not pass "
            "the safety check."
        )

    model = get_summary_model()

    prompt = SUMMARY_PROMPT.invoke(
        {
            "extracted_text":
                extracted_text,
        }
    )

    response = model.invoke(
        prompt
    )

    content = response.content

    if not isinstance(
        content,
        str,
    ):
        raise ValueError(
            "Summary model returned invalid content."
        )

    content = content.strip()

    if "TITLE:" not in content:
        raise ValueError(
            "Summary response does not contain "
            "a TITLE section."
        )

    if "SUMMARY:" not in content:
        raise ValueError(
            "Summary response does not contain "
            "a SUMMARY section."
        )

    title_part = content.split(
        "TITLE:",
        1,
    )[1]

    title_part = title_part.split(
        "SUMMARY:",
        1,
    )[0]

    generated_title = (
        title_part.strip()
    )

    generated_summary = (
        content.split(
            "SUMMARY:",
            1,
        )[1]
        .strip()
    )

    if not generated_title:
        raise ValueError(
            "Generated title is empty."
        )

    if not generated_summary:
        raise ValueError(
            "Generated summary is empty."
        )

    updated = update_memory_record(
        memory_id=memory_id,
        user_id=user_id,
        title=generated_title,
        summary=generated_summary,
    )

    if not updated:
        raise ValueError(
            "Memory could not be updated."
        )

    return get_memory(
        memory_id,
        user_id,
    )


