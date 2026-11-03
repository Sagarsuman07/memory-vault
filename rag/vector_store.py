"""Qdrant vector store for Memory Vault (dense vectors + BM25 sparse vectors).

Each chunk is one Qdrant point with two named vectors:

* ``dense`` - the existing all-MiniLM-L6-v2 embedding (cosine similarity).
* ``bm25``  - a sparse term-weight vector. Qdrant applies the IDF part of BM25
  server-side (``Modifier.IDF``); the term-frequency part is computed here.

Search (rag/retriever.py) combines both with Qdrant's native RRF fusion.

Public API kept from the Chroma version, so callers are unchanged:
``index_memory``, ``verify_memory_index``, ``clone_memory_index``,
``VectorReuseError``. New: ``delete_memory_index`` and ``get_client``.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
import zlib
from collections import Counter
from numbers import Real
from typing import Any, Callable, Iterable, Optional, Sequence

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from config.logging_config import StageTimer, get_logger, log_event, timed_lock
from config.settings import settings
from rag.embeddings import get_embeddings

logger = get_logger(__name__)

COLLECTION_NAME = settings.QDRANT_COLLECTION
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
TEXT_KEY = "chunk_text"          # payload key holding the chunk text
UPSERT_BATCH_SIZE = 64

# BM25 term-frequency parameters. AVG_DOC_LEN is the typical number of terms in
# one chunk (CHUNK_SIZE is ~1000 characters).
BM25_K1 = 1.2
BM25_B = 0.75
BM25_AVG_DOC_LEN = 170.0

# Serialize Qdrant writes and collection operations within this app process.
QDRANT_LOCK = threading.RLock()
_CLIENT_LOCK = threading.Lock()
_CLIENT: QdrantClient | None = None
_EMBEDDING_LOCK = threading.Semaphore(
    max(1, int(getattr(settings, "EMBEDDING_CONCURRENCY", 1)))
)


# ============================================================
# Client + collection
# ============================================================

def _ensure_collection(client: QdrantClient) -> None:
    """Create or validate the hybrid collection and its filter indexes."""
    size = settings.EMBEDDING_DIMENSION
    if client.collection_exists(COLLECTION_NAME):
        params = client.get_collection(COLLECTION_NAME).config.params
        dense = params.vectors.get(DENSE_VECTOR) if isinstance(params.vectors, dict) else None
        if dense is None or dense.size != size:
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not have a "
                f"'{DENSE_VECTOR}' vector of size {size}. Use another QDRANT_COLLECTION "
                "or migrate the existing collection."
            )
        if SPARSE_VECTOR not in (params.sparse_vectors or {}):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' has no '{SPARSE_VECTOR}' sparse vector."
            )
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=size, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )

    # Existing collections may predate these indexes. Ensure both filters
    # are indexed before the app starts serving user-scoped retrieval.
    payload_schema = client.get_collection(COLLECTION_NAME).payload_schema or {}
    for field in ("user_id", "memory_id"):
        if field not in payload_schema:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )


def get_client() -> QdrantClient:
    """Return the shared Qdrant Cloud client (created once, collection ensured)."""
    global _CLIENT
    if _CLIENT is None:
        with timed_lock(logger, _CLIENT_LOCK, "qdrant_client", "construct_client"):
            if _CLIENT is None:
                started = time.perf_counter()
                try:
                    settings.validate_qdrant_cloud_settings()
                    client = QdrantClient(
                        url=settings.QDRANT_URL,
                        api_key=settings.QDRANT_API_KEY,
                        timeout=30,
                    )
                    with QDRANT_LOCK:
                        _ensure_collection(client)
                except Exception as error:
                    log_event(
                        logger, "INGEST_COMPONENT_INITIALIZATION", logging.WARNING,
                        component="qdrant_client", status="error",
                        error=type(error).__name__,
                        seconds=time.perf_counter() - started,
                    )
                    raise
                _CLIENT = client
                log_event(
                    logger, "INGEST_COMPONENT_INITIALIZATION",
                    component="qdrant_client", status="success",
                    seconds=time.perf_counter() - started,
                )
    return _CLIENT


# ============================================================
# BM25 sparse vectors
# ============================================================

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its me my of on or "
    "that the this to was were what when where which who why with you your".split()
)


def query_terms(text: str) -> list[str]:
    """Lower-case content words (stop words removed); used for BM25 and grounding."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _term_id(term: str) -> int:
    # crc32 is stable across runs (Python's hash() is randomised) and fits Qdrant's u32 index.
    return zlib.crc32(term.encode("utf-8"))


def sparse_document(text: str) -> models.SparseVector:
    """BM25 term-frequency weights for a chunk (IDF is applied by Qdrant)."""
    terms = query_terms(text)
    counts = Counter(_term_id(t) for t in terms)
    length_norm = BM25_K1 * (1 - BM25_B + BM25_B * len(terms) / BM25_AVG_DOC_LEN)
    indices = sorted(counts)
    return models.SparseVector(
        indices=indices,
        values=[counts[i] * (BM25_K1 + 1) / (counts[i] + length_norm) for i in indices],
    )


def sparse_query(text: str) -> models.SparseVector | None:
    """Query vector (weight 1 per distinct term), or None if no content words."""
    indices = sorted({_term_id(t) for t in query_terms(text)})
    if not indices:
        return None
    return models.SparseVector(indices=indices, values=[1.0] * len(indices))


# ============================================================
# Helpers
# ============================================================

def memory_filter(memory_id: str, user_id: str | None = None) -> models.Filter:
    must = [models.FieldCondition(key="memory_id", match=models.MatchValue(value=memory_id))]
    if user_id is not None:
        must.append(models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)))
    return models.Filter(must=must)


def _point_id(memory_id: str, chunk_index: int) -> str:
    """Deterministic UUID (Qdrant IDs must be UUIDs/ints), so re-indexing overwrites."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{memory_id}_chunk_{chunk_index}"))


def _chunk_id(memory_id: str, chunk_index: int) -> str:
    return f"{memory_id}_chunk_{chunk_index}"


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Keep scalar metadata only."""
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items()
            if isinstance(k, str) and k.strip() and v is not None
            and isinstance(v, (str, bool, int, float))}


def _embed_batches(texts: Sequence[str], embedder: Any, batch_size: int,
                   existing_dimension: int | None = None) -> Iterable[tuple[int, list[list[float]]]]:
    """Yield validated embedding batches to keep peak vector memory bounded."""
    if batch_size <= 0:
        raise ValueError("EMBEDDING_BATCH_SIZE must be greater than zero.")
    if not texts:
        raise ValueError("Cannot generate embeddings for an empty text list.")

    dimension = existing_dimension
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        started = time.perf_counter()
        try:
            with timed_lock(logger, _EMBEDDING_LOCK, "embedding", "embed_batch",
                            batch=start // batch_size + 1, items=len(batch)):
                raw = embedder.embed_documents(batch)
        except Exception as error:
            log_event(logger, "INGEST_EMBEDDING_BATCH_ERROR", logging.WARNING,
                      batch=start // batch_size + 1, start=start, items=len(batch),
                      error=type(error).__name__, seconds=time.perf_counter() - started)
            raise

        vectors = _as_list(raw)
        if len(vectors) != len(batch):
            raise ValueError("Embedding response count does not match input batch.")
        validated = []
        for raw_vector in vectors:
            vector = _as_list(raw_vector)
            if not vector or any(
                isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v))
                for v in vector
            ):
                raise ValueError("Embedding vectors must contain finite numeric values.")
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {dimension}, received {len(vector)}."
                )
            validated.append([float(v) for v in vector])

        log_event(logger, "INGEST_EMBEDDING_BATCH_SUCCESS",
                  batch=start // batch_size + 1, start=start, items=len(batch),
                  seconds=time.perf_counter() - started)
        yield start, validated


def _iter_scroll(client: QdrantClient, flt: models.Filter, *,
                with_payload: bool = False,
                with_vectors: bool | list[str] = False):
    """Yield scroll pages so callers can avoid retaining every Qdrant point."""
    offset = None
    while True:
        batch, offset = client.scroll(
            COLLECTION_NAME, scroll_filter=flt, limit=256, offset=offset,
            with_payload=with_payload, with_vectors=with_vectors,
        )
        yield from batch
        if offset is None:
            break


def _scroll(client: QdrantClient, flt: models.Filter, *,
            with_payload: bool = False,
            with_vectors: bool | list[str] = False) -> list:
    return list(_iter_scroll(
        client, flt, with_payload=with_payload, with_vectors=with_vectors
    ))


def _make_point(memory_id: str, user_id: str, memory_type: str, index: int, text: str,
                metadata: dict[str, Any], dense: list[float]) -> models.PointStruct:
    payload = dict(metadata)
    # Identity fields are set last so chunk metadata can never override them.
    payload.update({"memory_id": memory_id, "user_id": user_id, "memory_type": memory_type,
                    "chunk_index": index, "chunk_id": _chunk_id(memory_id, index),
                    TEXT_KEY: text})
    return models.PointStruct(id=_point_id(memory_id, index), payload=payload,
                              vector={DENSE_VECTOR: dense, SPARSE_VECTOR: sparse_document(text)})


def _replace_memory_points(
    client: QdrantClient,
    memory_id: str,
    points: Iterable[models.PointStruct],
    expected_count: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Upsert bounded batches, then remove stale points for this memory."""
    started = time.perf_counter()
    with timed_lock(
        logger, QDRANT_LOCK, "qdrant", "replace_memory_index",
        memory=memory_id, chunks=expected_count,
    ):
        previous = {
            point.id for point in _iter_scroll(client, memory_filter(memory_id))
        }
        point_ids = set()
        batch = []
        try:
            for point in points:
                point_ids.add(point.id)
                batch.append(point)
                if len(batch) >= UPSERT_BATCH_SIZE:
                    if progress_callback:
                        progress_callback("indexing")
                        progress_callback = None
                    client.upsert(COLLECTION_NAME, points=batch, wait=True)
                    batch = []
            if batch:
                if progress_callback:
                    progress_callback("indexing")
                client.upsert(COLLECTION_NAME, points=batch, wait=True)

            if len(point_ids) != expected_count:
                raise RuntimeError(
                    f"Qdrant indexing produced {len(point_ids)} points; "
                    f"expected {expected_count}."
                )

            stale = list(previous - point_ids)
            if stale:
                client.delete(
                    COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=stale),
                    wait=True,
                )
            actual = client.count(
                COLLECTION_NAME,
                count_filter=memory_filter(memory_id),
                exact=True,
            ).count
            if actual != expected_count:
                raise RuntimeError(
                    f"Qdrant verification failed: expected {expected_count} "
                    f"chunks, found {actual}."
                )
            log_event(
                logger, "INGEST_TIMING", stage="qdrant_indexing",
                memory=memory_id, seconds=time.perf_counter() - started,
                chunks=expected_count, stale_records=len(stale),
            )
        except Exception as error:
            log_event(
                logger, "INGEST_QDRANT_INDEX_ERROR", logging.ERROR,
                memory=memory_id, error=type(error).__name__,
                chunks=len(point_ids), seconds=time.perf_counter() - started,
            )
            created = list(point_ids - previous)
            try:
                if created:
                    client.delete(
                        COLLECTION_NAME,
                        points_selector=models.PointIdsList(points=created),
                        wait=True,
                    )
            except Exception as cleanup_error:
                log_event(
                    logger, "INGEST_QDRANT_CLEANUP_ERROR", logging.CRITICAL,
                    memory=memory_id, error=type(cleanup_error).__name__,
                )
                raise RuntimeError(
                    "Indexing failed and cleanup also failed. Inspect Qdrant before retrying. "
                    f"Indexing error: {error}; cleanup error: {cleanup_error}"
                ) from error
            raise


# ============================================================
# Indexing
# ============================================================

def index_memory(memory_id: str, user_id: str, memory_type: str,
                 chunks: list[Document],
                 progress_callback: Optional[Callable[[str], None]] = None) -> int:
    """Embed and index chunks in bounded Qdrant upsert batches."""
    for name, value in (
        ("memory_id", memory_id), ("user_id", user_id), ("memory_type", memory_type)
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} cannot be empty.")
    if not chunks:
        raise ValueError("Cannot index a memory with no chunks.")

    texts = []
    for index, chunk in enumerate(chunks):
        text = getattr(chunk, "page_content", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {index} has empty or invalid text.")
        texts.append(text)

    batch_size = int(getattr(settings, "EMBEDDING_BATCH_SIZE", 32))
    client = get_client()
    embedder = get_embeddings()
    if progress_callback:
        progress_callback("embedding")

    def iter_points():
        for start, vectors in _embed_batches(
            texts, embedder, batch_size, settings.EMBEDDING_DIMENSION
        ):
            for offset, vector in enumerate(vectors):
                index = start + offset
                yield _make_point(
                    memory_id, user_id, memory_type, index, texts[index],
                    _safe_metadata(getattr(chunks[index], "metadata", None)), vector,
                )

    # Embeddings flow directly into the bounded Qdrant upsert buffer.
    _replace_memory_points(
        client, memory_id, iter_points(), expected_count=len(chunks),
        progress_callback=progress_callback,
    )
    return len(chunks)


def delete_memory_index(memory_id: str) -> None:
    """Remove every chunk of a memory (no-op if it has none)."""
    if not memory_id or not memory_id.strip():
        raise ValueError("Memory ID cannot be empty.")
    client = get_client()
    with timed_lock(logger, QDRANT_LOCK, "qdrant", "delete_memory_index", memory=memory_id):
        client.delete(COLLECTION_NAME,
                      points_selector=models.FilterSelector(filter=memory_filter(memory_id)),
                      wait=True)


def verify_memory_index(memory_id: str, expected_chunks: list[Document] | None = None) -> dict[str, Any]:
    """Verify Qdrant pages incrementally without retaining all vector payloads."""
    if not memory_id or not memory_id.strip():
        raise ValueError("Memory ID cannot be empty.")

    client = get_client()
    ids, seen_ids, seen_indices, dimensions, errors = [], set(), set(), set(), []
    record_count = 0
    with timed_lock(logger, QDRANT_LOCK, "qdrant", "verify_memory_index", memory=memory_id):
        for point in _iter_scroll(
            client, memory_filter(memory_id), with_payload=True,
            with_vectors=[DENSE_VECTOR],
        ):
            record_count += 1
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            ids.append(chunk_id)
            if chunk_id in seen_ids:
                errors.append("Duplicate record IDs were returned.")
            seen_ids.add(chunk_id)
            if payload.get("memory_id") != memory_id:
                errors.append("A record has an incorrect memory_id.")

            vector = _as_list((point.vector or {}).get(DENSE_VECTOR))
            dimensions.add(len(vector))
            if not vector:
                errors.append("An indexed vector is missing.")

            if expected_chunks is not None:
                index = payload.get("chunk_index")
                if not isinstance(index, int) or not 0 <= index < len(expected_chunks):
                    errors.append("Indexed chunk positions do not match expected chunks.")
                else:
                    if index in seen_indices:
                        errors.append("Duplicate chunk positions were returned.")
                    seen_indices.add(index)
                    if chunk_id != _chunk_id(memory_id, index):
                        errors.append(f"Indexed chunk ID mismatch for chunk {index}.")
                    if payload.get(TEXT_KEY) != expected_chunks[index].page_content:
                        errors.append(f"Indexed text mismatch for chunk {index}.")

    if record_count == 0:
        errors.append("No indexed records found for this memory.")
    if len(dimensions) > 1 or 0 in dimensions:
        errors.append("Indexed vectors have inconsistent dimensions.")
    if expected_chunks is not None:
        if record_count != len(expected_chunks):
            errors.append("Indexed record count does not match expected chunks.")
        if seen_indices != set(range(len(expected_chunks))):
            errors.append("Indexed chunk positions do not cover the expected range.")

    return {
        "valid": not errors, "memory_id": memory_id, "record_count": record_count,
        "embedding_dimension": next(iter(dimensions)) if len(dimensions) == 1 else None,
        "errors": list(dict.fromkeys(errors)), "ids": ids,
    }


# ============================================================
# Duplicate reuse
# ============================================================

class VectorReuseError(Exception):
    """The source memory's vectors cannot be safely reused (``reason`` says why)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def clone_memory_index(*, source_memory_id: str, target_memory_id: str, user_id: str,
                       memory_type: str, expected_texts: list[str]) -> int:
    """Copy an existing memory's dense vectors to a new memory without re-embedding.

    The source records must be complete, belong to ``user_id`` and their stored
    text must equal ``expected_texts`` (what the *current* chunking produces),
    otherwise ``VectorReuseError`` is raised and the caller falls back to the
    normal pipeline. The source is only ever read.
    """
    for name, value in (("source_memory_id", source_memory_id), ("target_memory_id", target_memory_id),
                        ("user_id", user_id), ("memory_type", memory_type)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} cannot be empty.")
    if source_memory_id == target_memory_id:
        raise VectorReuseError("source_is_target")
    if not expected_texts:
        raise VectorReuseError("no_expected_chunks")

    client = get_client()
    with timed_lock(logger, QDRANT_LOCK, "qdrant", "read_duplicate_source_vectors",
                    memory=source_memory_id, target_memory=target_memory_id):
        source = _scroll(client, memory_filter(source_memory_id, user_id), with_payload=True,
                         with_vectors=[DENSE_VECTOR])
    if not source:
        raise VectorReuseError("source_vectors_missing")
    if len(source) != len(expected_texts):
        raise VectorReuseError("chunk_count_mismatch")

    by_index = {(p.payload or {}).get("chunk_index"): p for p in source}
    if len(by_index) != len(source):
        raise VectorReuseError("source_duplicate_ids")

    points = []
    for index, expected_text in enumerate(expected_texts):
        source_point = by_index.get(index)
        if source_point is None:
            raise VectorReuseError("chunk_ids_mismatch")
        payload = dict(source_point.payload or {})
        if payload.get(TEXT_KEY) != expected_text:
            raise VectorReuseError("chunk_text_mismatch")
        if payload.get("memory_id") != source_memory_id or payload.get("user_id") != user_id:
            raise VectorReuseError("source_metadata_mismatch")
        dense = [float(x) for x in _as_list((source_point.vector or {}).get(DENSE_VECTOR))]
        if not dense or any(not math.isfinite(x) for x in dense):
            raise VectorReuseError("source_vector_invalid")
        if len(dense) != settings.EMBEDDING_DIMENSION:
            raise VectorReuseError("dimension_mismatch")
        for key in ("memory_id", "user_id", "memory_type", "chunk_index", "chunk_id", TEXT_KEY):
            payload.pop(key, None)
        points.append(_make_point(target_memory_id, user_id, memory_type, index,
                                  expected_text, payload, dense))

    _replace_memory_points(client, target_memory_id, points, expected_count=len(points))
    return len(points)