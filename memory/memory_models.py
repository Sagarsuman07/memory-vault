"""Shared, dependency-light contracts for Memory Vault preparation.

These types describe pipeline data and lifecycle states. They do not execute
pipeline work or mutate the database.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class MemoryStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PipelineStage(str, Enum):
    DISCOVERY = "discovery"
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETE = "complete"


class PreparationErrorCategory(str, Enum):
    VALIDATION = "validation"
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INTERNAL = "internal"


JOB_STATUS_TRANSITIONS: Mapping[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.FAILED}),
    JobStatus.RUNNING: frozenset(
        {JobStatus.RETRYING, JobStatus.SUCCEEDED, JobStatus.FAILED}
    ),
    JobStatus.RETRYING: frozenset({JobStatus.RUNNING, JobStatus.FAILED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset({JobStatus.RETRYING}),
}

MEMORY_STATUS_TRANSITIONS: Mapping[MemoryStatus, frozenset[MemoryStatus]] = {
    MemoryStatus.PENDING: frozenset({MemoryStatus.PROCESSING, MemoryStatus.FAILED}),
    MemoryStatus.PROCESSING: frozenset({MemoryStatus.READY, MemoryStatus.FAILED}),
    MemoryStatus.READY: frozenset({MemoryStatus.PROCESSING}),
    MemoryStatus.FAILED: frozenset({MemoryStatus.PROCESSING}),
}

STAGE_ORDER: tuple[PipelineStage, ...] = (
    PipelineStage.DISCOVERY,
    PipelineStage.EXTRACTION,
    PipelineStage.CHUNKING,
    PipelineStage.EMBEDDING,
    PipelineStage.INDEXING,
    PipelineStage.COMPLETE,
)


class InvalidStateTransition(ValueError):
    """Raised when a lifecycle state transition is not allowed."""


def validate_job_transition(current: JobStatus | str, target: JobStatus | str) -> None:
    current_status = JobStatus(current)
    target_status = JobStatus(target)
    if target_status not in JOB_STATUS_TRANSITIONS[current_status]:
        raise InvalidStateTransition(
            f"Invalid job status transition: {current_status.value} -> {target_status.value}"
        )


def validate_memory_transition(
    current: MemoryStatus | str, target: MemoryStatus | str
) -> None:
    current_status = MemoryStatus(current)
    target_status = MemoryStatus(target)
    if target_status not in MEMORY_STATUS_TRANSITIONS[current_status]:
        raise InvalidStateTransition(
            f"Invalid memory status transition: {current_status.value} -> {target_status.value}"
        )


@dataclass(frozen=True)
class SourceDocument:
    """A source file and extracted content passed between pipeline stages."""

    source_id: str
    memory_id: str
    file_name: str
    media_type: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extraction_status: str = "succeeded"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_id", "memory_id", "file_name", "media_type"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.extraction_status not in {"pending", "succeeded", "failed", "partial"}:
            raise ValueError("Unsupported extraction_status")
        if self.extraction_status == "succeeded" and not self.text.strip():
            raise ValueError("Successfully extracted document text must not be empty")


@dataclass(frozen=True)
class MemoryChunk:
    """A text chunk with stable identity and source-traceability metadata."""

    chunk_id: str
    memory_id: str
    source_id: str
    chunk_index: int
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("chunk_id", "memory_id", "source_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be zero or greater")
        if not self.text.strip():
            raise ValueError("chunk text must not be empty")


@dataclass(frozen=True)
class PreparationJob:
    """Persistable view of one preparation job and its latest progress."""

    id: str
    memory_id: str
    user_id: str
    status: JobStatus = JobStatus.QUEUED
    current_stage: PipelineStage = PipelineStage.DISCOVERY
    attempt_count: int = 0
    completed_files: int = 0
    total_files: int = 1
    error_category: Optional[PreparationErrorCategory] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("id", "memory_id", "user_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.attempt_count < 0:
            raise ValueError("attempt_count must be zero or greater")
        if self.total_files < 0 or self.completed_files < 0:
            raise ValueError("File counts must be zero or greater")
        if self.completed_files > self.total_files:
            raise ValueError("completed_files cannot exceed total_files")
        if self.status == JobStatus.SUCCEEDED and self.current_stage != PipelineStage.COMPLETE:
            raise ValueError("A succeeded job must have current_stage=complete")


@dataclass(frozen=True)
class StageResult:
    """Standard result envelope returned by a pipeline stage."""

    stage: PipelineStage
    succeeded: bool
    item_count: int = 0
    warnings: tuple[str, ...] = ()
    error_category: Optional[PreparationErrorCategory] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.item_count < 0:
            raise ValueError("item_count must be zero or greater")
        if self.succeeded and self.error_category is not None:
            raise ValueError("Successful stage results cannot have an error category")
        if not self.succeeded and self.error_category is None:
            raise ValueError("Failed stage results require an error category")
