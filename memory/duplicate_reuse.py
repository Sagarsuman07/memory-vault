"""Safe reuse of a completed memory when the same file is uploaded again.

``try_reuse_duplicate`` is called by the background worker before extraction.
If an eligible, fully indexed memory with identical file content (SHA-256)
already exists *for the same user*, its extracted text and vectors are reused:
no extraction/OCR, chunking-to-embedding, or embedding work is repeated.

If anything about the candidate is not safe, the function returns ``None`` and
the worker runs the existing normal ingestion pipeline. This module never
raises, never writes to SQLite, and never modifies the source memory.

Set ``DUPLICATE_REUSE_ENABLED=false`` in the environment to switch the feature
off without changing code.

Only identifiers, counts and timings are logged - never file content.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any

from config.logging_config import get_logger, log_event
from database.database import get_connection
from database.repositories import get_ingestion_source, get_memory

logger = get_logger(__name__)

_MAX_CANDIDATES = 10
SKIPPED_STAGES = "extraction,ocr,embedding"


@dataclass(frozen=True)
class ReuseResult:
    """Everything the worker needs to finish the job after vectors were cloned."""
    source_memory_id: str
    chunks: list            # list[Document] re-derived from the source's extracted text
    extracted_text: str
    metadata_json: str
    warnings_json: str


class _Ineligible(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def duplicate_reuse_enabled() -> bool:
    return os.getenv("DUPLICATE_REUSE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def _find_candidates(user_id: str, sha256: str, exclude_memory_id: str) -> tuple[list[dict[str, Any]], int]:
    """Same-user, same-hash sources marked loaded whose memory STILL EXISTS.

    Deleting a memory leaves its ``ingestion_sources`` row behind, so the join on
    ``memories`` is what keeps stale rows from being counted or crowding out a
    valid candidate. Oldest first. Returns ``(candidates, stale_count)``.
    """
    connection = get_connection()
    try:
        rows = connection.execute(
            """SELECT s.memory_id, s.source_id, s.extension, s.metadata_json, s.warnings_json
               FROM ingestion_sources s
               JOIN memories m ON m.id = s.memory_id AND m.user_id = s.user_id
               WHERE s.user_id = ? AND s.sha256 = ? AND s.status = 'loaded'
                 AND s.memory_id IS NOT NULL AND s.memory_id != ? AND s.source_id != ?
                 AND m.extracted_text IS NOT NULL AND TRIM(m.extracted_text) != ''
               ORDER BY s.created_at ASC LIMIT ?""",
            (user_id, sha256, exclude_memory_id, exclude_memory_id, _MAX_CANDIDATES),
        ).fetchall()
        total = connection.execute(
            """SELECT COUNT(*) AS n FROM ingestion_sources
               WHERE user_id = ? AND sha256 = ? AND status = 'loaded'
                 AND memory_id IS NOT NULL AND memory_id != ? AND source_id != ?""",
            (user_id, sha256, exclude_memory_id, exclude_memory_id),
        ).fetchone()["n"]
        candidates = [dict(row) for row in rows]
        return candidates, max(0, int(total) - len(candidates))
    finally:
        connection.close()


def _has_active_job(memory_id: str, user_id: str) -> bool:
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT 1 FROM preparation_jobs WHERE memory_id = ? AND user_id = ? "
            "AND status IN ('queued', 'running', 'retrying') LIMIT 1",
            (memory_id, user_id),
        ).fetchone()
        return row is not None
    finally:
        connection.close()


def _validate_and_clone(candidate: dict[str, Any], *, target_memory_id: str, user_id: str,
                        memory_type: str, extension: str) -> ReuseResult:
    from rag.chunker import split_text
    from rag.vector_store import clone_memory_index

    source_id = candidate["memory_id"]
    memory = get_memory(source_id, user_id)               # scoped to this user
    if memory is None:
        raise _Ineligible("candidate_memory_missing")
    text = memory["extracted_text"]
    if not isinstance(text, str) or not text.strip():
        raise _Ineligible("candidate_text_missing")
    if memory["memory_type"] != memory_type:
        raise _Ineligible("memory_type_mismatch")
    if (candidate.get("extension") or "").lower() != (extension or "").lower():
        raise _Ineligible("extension_mismatch")
    if _has_active_job(source_id, user_id):
        raise _Ineligible("candidate_job_active")

    try:
        metadata = json.loads(candidate.get("metadata_json") or "{}")
        if not isinstance(metadata, dict):
            metadata = {}
    except (TypeError, ValueError):
        metadata = {}

    # Re-chunk with the *current* settings; clone_memory_index requires the
    # stored vectors' text to match exactly, so stale/changed chunking is refused.
    chunks = split_text(text, metadata=metadata)
    if not chunks:
        raise _Ineligible("rechunk_empty")
    clone_started = time.perf_counter()
    cloned = clone_memory_index(
        source_memory_id=source_id,
        target_memory_id=target_memory_id,
        user_id=user_id,
        memory_type=memory_type,
        expected_texts=[chunk.page_content for chunk in chunks],
    )
    log_event(logger, "INGEST_TIMING", stage="duplicate_vector_clone", memory=target_memory_id,
              source_memory=source_id, seconds=time.perf_counter() - clone_started, chunks=cloned)
    return ReuseResult(
        source_memory_id=source_id,
        chunks=chunks,
        extracted_text=text,
        metadata_json=candidate.get("metadata_json") or "{}",
        warnings_json=candidate.get("warnings_json") or "[]",
    )


def try_reuse_duplicate(*, job_id: str, memory_id: str, user_id: str, memory_type: str,
                        file_label: str) -> ReuseResult | None:
    """Return a ``ReuseResult`` (vectors already cloned) or ``None`` to fall back."""
    started = time.perf_counter()
    common = {"job": job_id, "memory": memory_id, "file": file_label}
    try:
        if not duplicate_reuse_enabled():
            log_event(logger, "INGEST_DUPLICATE_FALLBACK", **common, reason="feature_disabled")
            return None

        source = get_ingestion_source(memory_id, user_id)
        if source is None or not source.sha256:
            log_event(logger, "INGEST_DUPLICATE_FALLBACK", **common, reason="no_source_record")
            return None

        candidates, stale = _find_candidates(user_id, source.sha256, memory_id)
        log_event(logger, "INGEST_DUPLICATE_CHECK", **common, sha256=source.sha256[:12],
                  result="found" if candidates else "not_found", candidates=len(candidates),
                  stale_sources_ignored=stale, seconds=time.perf_counter() - started)
        if not candidates:
            return None

        for candidate in candidates:
            reason = "unknown"
            try:
                result = _validate_and_clone(candidate, target_memory_id=memory_id, user_id=user_id,
                                             memory_type=memory_type, extension=source.extension)
                log_event(logger, "INGEST_DUPLICATE_REUSE", **common,
                          source_memory=result.source_memory_id, target_memory=memory_id,
                          chunks=len(result.chunks), skipped_stages=SKIPPED_STAGES,
                          seconds=time.perf_counter() - started)
                return result
            except _Ineligible as error:
                reason = error.reason
            except Exception as error:  # includes VectorReuseError
                reason = getattr(error, "reason", None) or f"error_{type(error).__name__}"
            log_event(logger, "INGEST_DUPLICATE_FALLBACK", logging.INFO, **common,
                      candidate_memory=candidate.get("memory_id"), reason=reason)

        log_event(logger, "INGEST_DUPLICATE_FALLBACK", **common, reason="no_eligible_candidate",
                  candidates=len(candidates), seconds=time.perf_counter() - started)
        return None
    except Exception as error:  # reuse must never break ingestion
        log_event(logger, "INGEST_DUPLICATE_FALLBACK", logging.WARNING, **common,
                  reason=f"unexpected_{type(error).__name__}")
        return None
