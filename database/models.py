from dataclasses import dataclass
from typing import Optional

from memory.memory_models import PreparationJob


@dataclass
class Memory:
    id: str
    user_id: str
    memory_type: str
    title: str
    file_name: str
    file_path: str
    summary: Optional[str]
    extracted_text: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class IngestionSource:
    """Persisted source-discovery record; independent from memory identity."""
    source_id: str
    user_id: str
    original_file_name: str
    extension: str
    media_type: str
    size_bytes: int
    sha256: str
    status: str
    memory_id: Optional[str] = None
    stored_file_path: Optional[str] = None
    duplicate_of_memory_id: Optional[str] = None
    warnings_json: str = "[]"
    metadata_json: str = "{}"
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


__all__ = ["Memory", "PreparationJob", "IngestionSource"]
