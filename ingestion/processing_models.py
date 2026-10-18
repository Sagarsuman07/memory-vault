"""Dependency-light result model for Phase 3 processing and extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ProcessedContent:
    """Cleaned content plus traceable extraction metadata and warnings."""
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    status: str = "succeeded"

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "partial", "failed"}:
            raise ValueError("status must be succeeded, partial, or failed")
        if self.status == "succeeded" and not self.text.strip():
            raise ValueError("Successfully processed content cannot be empty")
        if self.status == "partial" and not self.text.strip():
            raise ValueError("Partially processed content must contain usable text")


class ExtractionError(ValueError):
    """A source could not produce usable extracted content."""
