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


UPLOAD_DIR = Path(
    settings.UPLOAD_DIR
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
}


MAX_FILE_SIZE = 10 * 1024 * 1024


def validate_file(uploaded_file):

    extension = Path(
        uploaded_file.name
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise ValueError(
            "Only PDF and TXT files are supported."
        )

    if uploaded_file.size > MAX_FILE_SIZE:

        raise ValueError(
            "File size must be less than 10 MB."
        )


def create_new_memory(
    uploaded_file,
    title: str,
    user_id: str
):

    # -----------------------------------------
    # 1. Validate file
    # -----------------------------------------

    validate_file(
        uploaded_file
    )


    # -----------------------------------------
    # 2. Generate memory ID
    # -----------------------------------------

    memory_id = str(
        uuid.uuid4()
    )


    # -----------------------------------------
    # 3. Get original file information
    # -----------------------------------------

    original_file_name = uploaded_file.name

    extension = Path(
        original_file_name
    ).suffix.lower()


    # -----------------------------------------
    # 4. Determine memory type
    # -----------------------------------------

    memory_type = extension.replace(
        ".",
        ""
    )


    # -----------------------------------------
    # 5. Create file path
    # -----------------------------------------

    file_path = (
        UPLOAD_DIR
        / f"{memory_id}{extension}"
    )


    # -----------------------------------------
    # 6. Save original file
    # -----------------------------------------

    with open(
        file_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # -----------------------------------------
    # 7. Generate timestamps
    # -----------------------------------------

    now = datetime.now(
        timezone.utc
    ).isoformat()


    # -----------------------------------------
    # 8. Create Memory object
    # -----------------------------------------

    memory = Memory(
        id=memory_id,
        user_id=user_id,
        memory_type=memory_type,
        title=title,
        file_name=original_file_name,
        file_path=str(file_path),
        created_at=now,
        updated_at=now,
    )


    # -----------------------------------------
    # 9. Save metadata in database
    # -----------------------------------------

    try:

        create_memory(
            memory
        )

    except Exception:

        # Database insertion failed.
        # Remove the already saved file.

        if file_path.exists():

            file_path.unlink()

        raise


    return memory


def delete_memory(
    memory_id: str,
    user_id: str
):

    # -----------------------------------------
    # 1. Find memory
    # -----------------------------------------

    memory = get_memory(
        memory_id,
        user_id
    )


    if memory is None:

        return False


    # -----------------------------------------
    # 2. Delete physical file
    # -----------------------------------------

    file_path = Path(
        memory["file_path"]
    )


    if file_path.exists():

        file_path.unlink()


    # -----------------------------------------
    # 3. Delete database record
    # -----------------------------------------

    delete_memory_record(
        memory_id,
        user_id
    )


    return True