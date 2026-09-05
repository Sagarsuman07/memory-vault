from datetime import datetime, timezone
from pathlib import Path
import uuid

from config.settings import settings

from database.models import Memory

from database.repositories import (
    create_memory,
    get_memory,
    delete_memory as delete_memory_record,
)

from ingestion.pipeline import (
    extract_content,
    get_memory_type,
)


UPLOAD_DIR = Path(settings.UPLOAD_DIR)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


MAX_FILE_SIZE = 10 * 1024 * 1024


def validate_file(uploaded_file):
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


def create_new_memory(
    uploaded_file,
    title: str,
    user_id: str,
):
    # 1. Validate uploaded file.
    validate_file(uploaded_file)

    # 2. Generate unique memory ID.
    memory_id = str(uuid.uuid4())

    # 3. Preserve original file name.
    original_file_name = uploaded_file.name

    extension = Path(
        original_file_name
    ).suffix.lower()

    # 4. Determine canonical memory type.
    #
    # Example:
    # .pdf → document
    # .png → image
    # .mp3 → audio
    memory_type = get_memory_type(
        original_file_name
    )

    # 5. Store original file using memory ID.
    file_path = (
        UPLOAD_DIR
        / f"{memory_id}{extension}"
    )

    try:
        with open(file_path, "wb") as file:
            file.write(
                uploaded_file.getbuffer()
            )

        # 6. Extract content.
        #
        # This is the same flow for every memory.
        #
        # Document → parser
        # Image    → Ollama Cloud
        # Audio    → Groq Whisper
        extracted_text = extract_content(
            str(file_path)
        )

        # 7. Make sure extraction produced content.
        if not extracted_text.strip():
            raise ValueError(
                "No content could be extracted from "
                "the uploaded file."
            )

        # 8. Create timestamps.
        now = datetime.now(
            timezone.utc
        ).isoformat()

        # 9. Create memory object.
        memory = Memory(
            id=memory_id,
            user_id=user_id,
            memory_type=memory_type,
            title=title,
            file_name=original_file_name,
            file_path=str(file_path),
            extracted_text=extracted_text,
            created_at=now,
            updated_at=now,
        )

        # 10. Save metadata + extracted text.
        create_memory(memory)

        return memory

    except Exception:
        # If extraction or database creation fails,
        # remove the original file so we don't leave
        # an incomplete memory behind.
        if file_path.exists():
            file_path.unlink()

        raise


def delete_memory(
    memory_id: str,
    user_id: str,
):
    memory = get_memory(
        memory_id,
        user_id,
    )

    if memory is None:
        return False

    file_path = Path(
        memory["file_path"]
    )

    if file_path.exists():
        file_path.unlink()

    delete_memory_record(
        memory_id,
        user_id,
    )

    return True