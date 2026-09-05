from pathlib import Path

from ingestion.document_processor import (
    extract_document_text,
)

from ingestion.image_processor import (
    extract_text_from_image,
)

from ingestion.audio_processor import (
    extract_text_from_audio,
)


DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".mpeg",
    ".mpga",
    ".webm",
    ".ogg",
    ".flac",
}


def get_memory_type(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension in DOCUMENT_EXTENSIONS:
        return "document"

    if extension in IMAGE_EXTENSIONS:
        return "image"

    if extension in AUDIO_EXTENSIONS:
        return "audio"

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def extract_content(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension in DOCUMENT_EXTENSIONS:
        return extract_document_text(file_path)

    if extension in IMAGE_EXTENSIONS:
        return extract_text_from_image(file_path)

    if extension in AUDIO_EXTENSIONS:
        return extract_text_from_audio(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )