"""Phase 2 source discovery, validation, hashing, and change detection.

This module is intentionally independent of Streamlit, database repositories,
and model APIs so its behavior can be tested without external services.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional
from uuid import uuid4
import io
import zipfile


DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"})
SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024


class SourceValidationError(ValueError):
    """A source cannot safely enter the ingestion pipeline."""


@dataclass(frozen=True)
class SourceDescriptor:
    source_id: str
    file_name: str
    extension: str
    media_type: str
    file_path: Optional[str]
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        for field_name in ("source_id", "file_name", "extension", "media_type", "sha256"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be greater than zero")


@dataclass(frozen=True)
class SourceChange:
    source_id: str
    previous_sha256: str
    current_sha256: str

    @property
    def changed(self) -> bool:
        return self.previous_sha256 != self.current_sha256


def get_memory_type(file_name: str) -> str:
    extension = Path(file_name).suffix.lower()
    if extension in DOCUMENT_EXTENSIONS:
        return "document"
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    raise SourceValidationError(f"Unsupported file type: {extension or '(no extension)'}")


def get_media_type(file_name: str) -> str:
    extension = Path(file_name).suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp3": "audio/mpeg",
        ".mpeg": "audio/mpeg",
        ".mpga": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    if extension not in media_types:
        raise SourceValidationError(f"Unsupported file type: {extension or '(no extension)'}")
    return media_types[extension]


def _validate_signature(extension: str, data: bytes, file_name: str) -> None:
    """Perform lightweight integrity checks where formats have reliable markers."""
    if extension == ".pdf" and not data.startswith(b"%PDF-"):
        raise SourceValidationError("The PDF file does not have a valid PDF signature.")
    if extension in {".jpg", ".jpeg"} and not data.startswith(b"\xff\xd8\xff"):
        raise SourceValidationError("The image does not have a valid JPEG signature.")
    if extension == ".png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SourceValidationError("The image does not have a valid PNG signature.")
    if extension == ".gif" and not data.startswith((b"GIF87a", b"GIF89a")):
        raise SourceValidationError("The image does not have a valid GIF signature.")
    if extension == ".webp" and not (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
        raise SourceValidationError("The image does not have a valid WebP signature.")
    if extension == ".wav" and not (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"):
        raise SourceValidationError("The audio file does not have a valid WAV signature.")
    if extension == ".flac" and not data.startswith(b"fLaC"):
        raise SourceValidationError("The audio file does not have a valid FLAC signature.")
    if extension == ".ogg" and not data.startswith(b"OggS"):
        raise SourceValidationError("The audio file does not have a valid Ogg signature.")
    if extension in {".m4a"} and not (len(data) >= 12 and data[4:8] == b"ftyp"):
        raise SourceValidationError("The audio file does not have a valid MP4/M4A signature.")
    if extension == ".webm" and not data.startswith(b"\x1a\x45\xdf\xa3"):
        raise SourceValidationError("The audio file does not have a valid WebM/EBML signature.")
    if extension in {".mp3", ".mpeg", ".mpga"}:
        has_id3 = data.startswith(b"ID3")
        has_frame_sync = len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
        if not (has_id3 or has_frame_sync):
            raise SourceValidationError("The audio file does not have a recognizable MPEG audio signature.")
    if extension == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise SourceValidationError("The DOCX file is missing required document parts.")
                if archive.testzip() is not None:
                    raise SourceValidationError("The DOCX archive contains a corrupted file entry.")
        except SourceValidationError:
            raise
        except (zipfile.BadZipFile, OSError) as error:
            raise SourceValidationError("The DOCX file is not a valid Office document.") from error


def discover_upload(
    file_name: str,
    data: bytes,
    *,
    upload_dir: str | Path,
    source_id: Optional[str] = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> SourceDescriptor:
    """Validate an uploaded filename and its bytes before writing it to disk."""
    if not isinstance(file_name, str) or not file_name.strip():
        raise SourceValidationError("The uploaded file must have a filename.")
    # Prevent path traversal and keep only the user-supplied basename.
    safe_name = Path(file_name.replace("\\", "/")).name
    if safe_name in {"", ".", ".."}:
        raise SourceValidationError("The uploaded filename is invalid.")
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise SourceValidationError(f"Unsupported file type: {extension or '(no extension)'}")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise SourceValidationError("Uploaded file content must be bytes.")
    file_bytes = bytes(data)
    if not file_bytes:
        raise SourceValidationError("The selected file is empty.")
    if max_file_size <= 0:
        raise ValueError("max_file_size must be greater than zero")
    if len(file_bytes) > max_file_size:
        raise SourceValidationError(f"File size cannot exceed {max_file_size // (1024 * 1024)} MB.")
    _validate_signature(extension, file_bytes, safe_name)
    source_id = source_id or str(uuid4())
    target_path = Path(upload_dir) / f"{source_id}{extension}"
    return SourceDescriptor(
        source_id=source_id,
        file_name=safe_name,
        extension=extension,
        media_type=get_media_type(safe_name),
        file_path=str(target_path),
        size_bytes=len(file_bytes),
        sha256=sha256(file_bytes).hexdigest(),
    )


def discover_file(
    file_path: str | Path,
    *,
    source_id: Optional[str] = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> SourceDescriptor:
    """Discover and validate an existing file path without modifying it."""
    path = Path(file_path)
    if not path.exists():
        raise SourceValidationError("The source file does not exist.")
    if not path.is_file():
        raise SourceValidationError("The source path is not a regular file.")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise SourceValidationError("The source file cannot be read.") from error
    descriptor = discover_upload(
        path.name, data, upload_dir=path.parent, source_id=source_id,
        max_file_size=max_file_size,
    )
    # Preserve the actual existing path; discover_upload computes the standard upload path.
    return SourceDescriptor(
        source_id=descriptor.source_id,
        file_name=descriptor.file_name,
        extension=descriptor.extension,
        media_type=descriptor.media_type,
        file_path=str(path.resolve()),
        size_bytes=descriptor.size_bytes,
        sha256=descriptor.sha256,
    )


def detect_source_change(previous_sha256: str, current_file_path: str | Path, source_id: str) -> SourceChange:
    """Compare a stored source fingerprint with the current file contents."""
    if not previous_sha256 or len(previous_sha256) != 64:
        raise ValueError("previous_sha256 must be a SHA-256 hex digest")
    descriptor = discover_file(current_file_path, source_id=source_id)
    return SourceChange(
        source_id=source_id,
        previous_sha256=previous_sha256.lower(),
        current_sha256=descriptor.sha256,
    )
