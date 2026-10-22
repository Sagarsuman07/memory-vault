"""Compatibility entry point for discovery and processed-content extraction."""
from pathlib import Path

from ingestion.discovery import (
    AUDIO_EXTENSIONS, DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS,
    SourceChange, SourceDescriptor, SourceValidationError, detect_source_change,
    discover_file, discover_upload, get_memory_type, get_media_type,
)
from ingestion.processing import process_content


def extract_content(file_path: str) -> str:
    """Backward-compatible API: return cleaned extracted text as a string."""
    return process_content(file_path).text


__all__ = [
    "AUDIO_EXTENSIONS", "DOCUMENT_EXTENSIONS", "IMAGE_EXTENSIONS", "SUPPORTED_EXTENSIONS",
    "SourceChange", "SourceDescriptor", "SourceValidationError", "detect_source_change",
    "discover_file", "discover_upload", "extract_content", "get_memory_type", "get_media_type",
    "process_content",
]
