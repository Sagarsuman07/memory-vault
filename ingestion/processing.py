"""Shared content cleaning, Unicode normalization, and extraction validation."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from ingestion.processing_models import ExtractionError, ProcessedContent


def clean_and_normalize_text(text: str) -> str:
    """Normalize representation without rewriting semantic content.

    Dates, identifiers, punctuation, negations, prices, and table delimiters are
    deliberately left unchanged. Only encoding/whitespace artifacts are cleaned.
    """
    if not isinstance(text, str):
        raise TypeError("Extracted content must be a string")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\f", "\n")
    # Trim line-end artifacts while preserving deliberate internal spacing.
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_processed_content(result: ProcessedContent, *, source_name: str = "source") -> ProcessedContent:
    """Validate useful text and update only the result envelope, not its meaning."""
    cleaned = clean_and_normalize_text(result.text)
    if not cleaned:
        raise ExtractionError(f"No usable text could be extracted from {source_name}.")
    metadata: dict[str, Any] = dict(result.metadata)
    metadata.setdefault("character_count", len(cleaned))
    metadata.setdefault("line_count", len(cleaned.splitlines()))
    status = result.status
    if result.warnings and status == "succeeded":
        status = "partial"
    return ProcessedContent(
        text=cleaned,
        metadata=metadata,
        warnings=tuple(result.warnings),
        status=status,
    )


def process_content(file_path: str | Path) -> ProcessedContent:
    """Extract, clean, normalize, and validate a supported source file."""
    from ingestion.discovery import DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS, AUDIO_EXTENSIONS
    from ingestion.document_processor import extract_document_content
    from ingestion.image_processor import extract_image_content
    from ingestion.audio_processor import extract_audio_content

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Source file does not exist or is not a regular file.")
    extension = path.suffix.lower()
    if extension in DOCUMENT_EXTENSIONS:
        result = extract_document_content(str(path))
    elif extension in IMAGE_EXTENSIONS:
        result = extract_image_content(str(path))
    elif extension in AUDIO_EXTENSIONS:
        result = extract_audio_content(str(path))
    else:
        raise ExtractionError(f"Unsupported source type: {extension or '(no extension)'}")
    return validate_processed_content(result, source_name=path.name)
