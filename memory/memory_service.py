from datetime import datetime, timezone
from pathlib import Path
import uuid

from config.settings import settings

from database.models import Memory

from database.repositories import (
    create_memory,
    get_memory,
    get_all_memories,
    update_memory as update_memory_record,
    delete_memory as delete_memory_record,
)

from ingestion.pipeline import (
    extract_content,
    get_memory_type,
)


# ============================================================
# Configuration
# ============================================================

UPLOAD_DIR = Path(
    settings.UPLOAD_DIR
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ALLOWED_EXTENSIONS = {

    # Documents
    ".pdf",
    ".docx",
    ".txt",

    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",

    # Audio
    ".mp3",
    ".wav",
    ".m4a",
    ".mpeg",
    ".mpga",
    ".webm",
    ".ogg",
    ".flac",
}


MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# Validation
# ============================================================

def validate_file(
    uploaded_file
):

    extension = Path(
        uploaded_file.name
    ).suffix.lower()


    if extension not in ALLOWED_EXTENSIONS:

        raise ValueError(
            "Unsupported file type. "
            "Supported files are PDF, DOCX, TXT, "
            "JPG, JPEG, PNG, WEBP, GIF, MP3, WAV, "
            "M4A, MPEG, MPGA, WEBM, OGG and FLAC."
        )


    if uploaded_file.size > MAX_FILE_SIZE:

        raise ValueError(
            "File size must be less than 10 MB."
        )


# ============================================================
# CREATE MEMORY
# ============================================================

def create_new_memory(
    uploaded_file,
    title: str,
    user_id: str,
):

    # -----------------------------------------
    # 1. Validate file
    # -----------------------------------------

    validate_file(
        uploaded_file
    )


    # -----------------------------------------
    # 2. Validate title
    # -----------------------------------------

    if not title.strip():

        raise ValueError(
            "Memory title cannot be empty."
        )


    # -----------------------------------------
    # 3. Generate memory ID
    # -----------------------------------------

    memory_id = str(
        uuid.uuid4()
    )


    # -----------------------------------------
    # 4. Original file information
    # -----------------------------------------

    original_file_name = (
        uploaded_file.name
    )


    extension = Path(
        original_file_name
    ).suffix.lower()


    # -----------------------------------------
    # 5. Determine memory type
    # -----------------------------------------

    memory_type = get_memory_type(
        original_file_name
    )


    # -----------------------------------------
    # 6. Create storage path
    # -----------------------------------------

    file_path = (
        UPLOAD_DIR
        / f"{memory_id}{extension}"
    )


    try:

        # -------------------------------------
        # 7. Save original file
        # -------------------------------------

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )


        # -------------------------------------
        # 8. Extract content
        # -------------------------------------

        extracted_text = extract_content(
            str(file_path)
        )


        # -------------------------------------
        # 9. Validate extracted content
        # -------------------------------------

        if not extracted_text:

            raise ValueError(
                "No content could be extracted "
                "from the uploaded file."
            )


        if not extracted_text.strip():

            raise ValueError(
                "No readable content was extracted "
                "from the uploaded file."
            )


        # -------------------------------------
        # 10. Generate timestamps
        # -------------------------------------

        now = datetime.now(
            timezone.utc
        ).isoformat()


        # -------------------------------------
        # 11. Create Memory object
        # -------------------------------------

        memory = Memory(
            id=memory_id,
            user_id=user_id,
            memory_type=memory_type,
            title=title.strip(),
            file_name=original_file_name,
            file_path=str(file_path),
            summary=None,
            extracted_text=extracted_text,
            created_at=now,
            updated_at=now,
        )


        # -------------------------------------
        # 12. Save memory
        # -------------------------------------

        create_memory(
            memory
        )


        return memory


    except Exception:

        # -------------------------------------
        # Cleanup if anything fails
        # -------------------------------------

        if file_path.exists():

            file_path.unlink()

        raise


# ============================================================
# GET MEMORY
# ============================================================

def get_memory_by_id(
    memory_id: str,
    user_id: str
):

    memory = get_memory(
        memory_id,
        user_id
    )


    if memory is None:

        raise ValueError(
            "Memory not found."
        )


    return memory


# ============================================================
# LIST MEMORIES
# ============================================================

def list_memories(
    user_id: str
):

    return get_all_memories(
        user_id
    )


# ============================================================
# UPDATE MEMORY
# ============================================================

def update_memory(
    memory_id: str,
    user_id: str,
    title: str,
    summary: str
):

    if not title.strip():

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
        user_id
    )


# ============================================================
# DELETE MEMORY
# ============================================================

def delete_memory(
    memory_id: str,
    user_id: str
):

    # -----------------------------------------
    # 1. Get memory
    # -----------------------------------------

    memory = get_memory(
        memory_id,
        user_id
    )


    if memory is None:

        return False


    # -----------------------------------------
    # 2. Delete original file
    # -----------------------------------------

    file_path = Path(
        memory["file_path"]
    )


    if file_path.exists():

        file_path.unlink()


    # -----------------------------------------
    # 3. Delete database record
    # -----------------------------------------

    deleted = delete_memory_record(
        memory_id,
        user_id
    )


    return deleted