"""Structure-aware, configurable text chunking for Memory Vault."""
from __future__ import annotations

import re
from typing import Any, Mapping

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


_PAGE_MARKER = re.compile(r"^\s*\[Page\s+(\d+)\]\s*$", re.IGNORECASE)
_HEADING_MARKER = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def _is_chroma_scalar(value: Any) -> bool:
    """Return whether a value is safe for Chroma metadata."""
    return isinstance(value, (str, bool, int, float))


def _scalar_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep scalar source metadata; nested extraction details stay in SQLite."""
    if not metadata:
        return {}
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip() or value is None:
            continue
        if _is_chroma_scalar(value):
            result[key] = value
    return result


def _structure_segments(text: str) -> list[tuple[str, int | None, str | None]]:
    """Split input into page/section units without merging their boundaries."""
    segments: list[tuple[str, int | None, str | None]] = []
    current_lines: list[str] = []
    current_page: int | None = None
    current_section: str | None = None

    def flush() -> None:
        if current_lines and any(line.strip() for line in current_lines):
            segments.append(("\n".join(current_lines).strip(), current_page, current_section))
        current_lines.clear()

    for line in text.splitlines():
        page_match = _PAGE_MARKER.match(line)
        heading_match = _HEADING_MARKER.match(line)

        if page_match:
            flush()
            current_page = int(page_match.group(1))
            # Page markers are represented as metadata rather than searchable text.
            continue

        if heading_match:
            flush()
            current_section = heading_match.group(1).strip()
            current_lines.append(line)
            continue

        current_lines.append(line)

    flush()
    return segments


def split_text(
    text: str,
    metadata: Mapping[str, Any] | None = None,
) -> list[Document]:
    """Split extracted text into bounded chunks with source context.

    Existing callers can continue calling ``split_text(text)``. Optional
    extraction metadata is copied only when its values are scalar, because
    Chroma metadata does not accept nested dictionaries/lists. Page and section
    context is inferred from the existing ``[Page N]`` and Markdown heading
    markers emitted by the Phase 3 extractors.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP
    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        strip_whitespace=True,
    )
    base_metadata = _scalar_metadata(metadata)
    chunks: list[Document] = []
    seen_content: dict[str, int] = {}

    for segment_text, page_number, section_title in _structure_segments(text):
        if not segment_text.strip():
            continue
        segment_metadata = dict(base_metadata)
        if page_number is not None:
            segment_metadata["source_page"] = page_number
        if section_title:
            segment_metadata["section_title"] = section_title

        for chunk in splitter.create_documents([segment_text], metadatas=[segment_metadata]):
            content = chunk.page_content.strip()
            if not content:
                continue
            if len(content) > chunk_size:
                raise ValueError(
                    f"Chunk length {len(content)} exceeds configured CHUNK_SIZE {chunk_size}."
                )
            # Flag exact repeated chunks for diagnostics, but do not silently
            # drop them: repeated text can be meaningful and must remain traceable.
            occurrence = seen_content.get(content, 0)
            if occurrence:
                chunk.metadata["duplicate_content"] = True
                chunk.metadata["duplicate_occurrence"] = occurrence + 1
            seen_content[content] = occurrence + 1

            # Defensive filtering also protects Chroma when callers provide
            # custom metadata or when the splitter copies metadata through.
            chunk_metadata = _scalar_metadata(chunk.metadata)
            chunks.append(Document(page_content=content, metadata=chunk_metadata))

    return chunks
