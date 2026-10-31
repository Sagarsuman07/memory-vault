"""Safe, batched, verifiable Chroma indexing for Memory Vault."""
from __future__ import annotations
import logging
import math
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import threading
import time
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import settings
from rag.embeddings import get_embeddings
from config.logging_config import StageTimer, get_logger, log_event, timed_lock

logger = get_logger(__name__)

CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "memory_vault"

# One lock protects the shared Chroma collection across worker threads. Keep it
# process-wide; a lock created per job would not coordinate concurrent writes.
_CHROMA_LOCK = threading.RLock()
# Building a Chroma client concurrently on first use can crash inside Chroma
# (e.g. KeyError / "Could not connect to tenant"). Serialize construction only.
_CLIENT_LOCK = threading.Lock()
_EMBEDDING_LOCK = threading.Semaphore(
    max(1, int(getattr(settings, "EMBEDDING_CONCURRENCY", 1)))
)


def get_vector_store():
    """Return the existing persistent Chroma vector store."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    with timed_lock(logger, _CLIENT_LOCK, "chroma_client", "construct_client"):
        started = time.perf_counter()
        try:
            store = Chroma(collection_name=COLLECTION_NAME,
                           embedding_function=get_embeddings(),
                           persist_directory=str(CHROMA_DIR))
        except Exception as error:
            log_event(logger, "INGEST_COMPONENT_INITIALIZATION", logging.WARNING,
                      component="chroma_client", status="error", error=type(error).__name__,
                      seconds=time.perf_counter() - started)
            raise
        log_event(logger, "INGEST_COMPONENT_INITIALIZATION", component="chroma_client",
                  status="success", seconds=time.perf_counter() - started)
        return store


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _embed_batches(texts: Sequence[str], embedder: Any, batch_size: int,
                   existing_dimension: int | None = None) -> list[list[float]]:
    """Generate embeddings in bounded batches and validate every vector."""
    if batch_size <= 0:
        raise ValueError("EMBEDDING_BATCH_SIZE must be greater than zero.")
    if not texts:
        raise ValueError("Cannot generate embeddings for an empty text list.")
    result: list[list[float]] = []
    dimension = existing_dimension
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        batch_started = time.perf_counter()
        batch_number = start // batch_size + 1
        # Release the shared embedding slot after each batch, so a small memory
        # can make progress between batches of a large memory.
        try:
            with timed_lock(logger, _EMBEDDING_LOCK, "embedding", "embed_batch",
                             batch=batch_number, items=len(batch)):
                raw = embedder.embed_documents(batch)
        except Exception as error:
            log_event(logger, "INGEST_EMBEDDING_BATCH_ERROR", logging.WARNING,
                      batch=batch_number, start=start, items=len(batch),
                      error=type(error).__name__, seconds=time.perf_counter() - batch_started)
            raise
        log_event(logger, "INGEST_EMBEDDING_BATCH_SUCCESS", batch=batch_number,
                  start=start, items=len(batch), seconds=time.perf_counter() - batch_started)
        vectors = _as_list(raw)
        if len(vectors) != len(batch):
            raise ValueError("Embedding response count does not match input batch.")
        for raw_vector in vectors:
            vector = _as_list(raw_vector)
            if not vector:
                raise ValueError("Embedding provider returned an empty vector.")
            if any(isinstance(v, bool) or not isinstance(v, Real)
                   or not math.isfinite(float(v)) for v in vector):
                raise ValueError("Embedding vectors must contain only finite numeric values.")
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {dimension}, received {len(vector)}.")
            result.append([float(v) for v in vector])
    return result


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Keep Chroma-compatible scalar metadata only."""
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items()
            if isinstance(k, str) and k.strip() and v is not None
            and isinstance(v, (str, bool, int, float))}


def _snapshot(collection: Any, memory_id: str) -> dict[str, list]:
    """Snapshot current records so a failed replacement can roll back."""
    data = collection.get(where={"memory_id": memory_id},
                          include=["embeddings", "documents", "metadatas"])
    ids = list(data.get("ids") or [])
    if not ids:
        return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
    embeddings = _as_list(data.get("embeddings"))
    documents = list(data.get("documents") or [])
    metadatas = list(data.get("metadatas") or [])
    if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
        raise RuntimeError("Cannot safely replace memory: existing Chroma records could not be snapshotted completely.")
    if any(not _as_list(v) for v in embeddings):
        raise RuntimeError("Cannot safely replace memory: an existing vector is missing.")
    return {"ids": ids,
            "embeddings": [[float(x) for x in _as_list(v)] for v in embeddings],
            "documents": documents, "metadatas": [dict(m or {}) for m in metadatas]}


def _collection_dimension(collection: Any) -> int | None:
    data = collection.get(limit=1, include=["embeddings"])
    vectors = _as_list(data.get("embeddings"))
    return len(_as_list(vectors[0])) if vectors and _as_list(vectors[0]) else None


def _verify_records(collection: Any, memory_id: str, ids: list[str],
                    texts: list[str], metadatas: list[dict], dimension: int) -> None:
    data = collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])
    found = list(data.get("ids") or [])
    if set(found) != set(ids) or len(found) != len(ids):
        raise RuntimeError("Chroma verification failed: indexed record IDs are incomplete or duplicated.")
    docs = list(data.get("documents") or [])
    metas = list(data.get("metadatas") or [])
    vectors = _as_list(data.get("embeddings"))
    if not (len(found) == len(docs) == len(metas) == len(vectors)):
        raise RuntimeError("Chroma verification failed: record fields are incomplete.")
    expected = {record_id: i for i, record_id in enumerate(ids)}
    for i, record_id in enumerate(found):
        j = expected[record_id]
        if docs[i] != texts[j]:
            raise RuntimeError(f"Chroma verification failed: text mismatch for {record_id}.")
        if dict(metas[i] or {}) != metadatas[j]:
            raise RuntimeError(f"Chroma verification failed: metadata mismatch for {record_id}.")
        if len(_as_list(vectors[i])) != dimension:
            raise RuntimeError(f"Chroma verification failed: vector dimension mismatch for {record_id}.")
        if (metas[i] or {}).get("memory_id") != memory_id:
            raise RuntimeError(f"Chroma verification failed: memory identity mismatch for {record_id}.")


def _rollback(collection: Any, memory_id: str, old: dict[str, list],
              attempted_ids: list[str]) -> None:
    """Restore the prior records and remove newly created partial records."""
    old_ids = set(old["ids"])
    new_ids = [x for x in attempted_ids if x not in old_ids]
    if new_ids:
        collection.delete(ids=new_ids)
    if old["ids"]:
        collection.upsert(ids=old["ids"], embeddings=old["embeddings"],
                          documents=old["documents"], metadatas=old["metadatas"])
    current = collection.get(where={"memory_id": memory_id}, include=["metadatas"])
    stale = [x for x in (current.get("ids") or []) if x not in old_ids]
    if stale:
        collection.delete(ids=stale)


def index_memory(memory_id: str, user_id: str, memory_type: str,
                 chunks: list[Document],
                 progress_callback: Optional[Callable[[str], None]] = None) -> int:
    """Embed independently, then serialize the Chroma replace/verify transaction."""
    if not memory_id or not memory_id.strip():
        raise ValueError("Memory ID cannot be empty.")
    if not user_id or not user_id.strip():
        raise ValueError("User ID cannot be empty.")
    if not memory_type or not memory_type.strip():
        raise ValueError("Memory type cannot be empty.")
    if not chunks:
        raise ValueError("Cannot index a memory with no chunks.")

    documents, ids = [], []
    for i, chunk in enumerate(chunks):
        text = getattr(chunk, "page_content", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {i} has empty or invalid text.")
        metadata = _safe_metadata(getattr(chunk, "metadata", None))
        metadata.update({"memory_id": memory_id, "user_id": user_id,
                         "memory_type": memory_type, "chunk_index": i})
        documents.append(Document(page_content=text, metadata=metadata))
        ids.append(f"{memory_id}_chunk_{i}")

    texts = [d.page_content for d in documents]
    metadatas = [dict(d.metadata) for d in documents]
    batch_size = int(getattr(settings, "EMBEDDING_BATCH_SIZE", 32))
    if batch_size <= 0:
        raise ValueError("EMBEDDING_BATCH_SIZE must be greater than zero.")

    # Read the established collection dimension while protected from concurrent
    # collection operations. Do not hold this lock during potentially slow model
    # inference; different jobs can prepare their embeddings independently.
    collection = get_vector_store()._collection
    with timed_lock(logger, _CHROMA_LOCK, "chroma", "read_collection_dimension",
                    memory=memory_id):
        existing_dimension = _collection_dimension(collection)

    if progress_callback:
        progress_callback("embedding")
    # The supplied rag/embeddings.py uses HuggingFaceEmbeddings locally, not an
    # embedding API. _embed_batches limits local inference per batch to reduce
    # CPU/RAM spikes while allowing other jobs between batches.
    embed_started = time.perf_counter()
    vectors = _embed_batches(texts, get_embeddings(), batch_size, existing_dimension)
    dimension = len(vectors[0])
    log_event(logger, "INGEST_TIMING", stage="embedding", memory=memory_id,
              seconds=time.perf_counter() - embed_started, chunks=len(texts), dimension=dimension)

    if progress_callback:
        progress_callback("indexing")
    # Snapshot + upsert + verify + stale cleanup + rollback are one critical
    # section so two jobs cannot interleave writes to the shared collection.
    chroma_started = time.perf_counter()
    with timed_lock(logger, _CHROMA_LOCK, "chroma", "replace_memory_index",
                    memory=memory_id, chunks=len(ids)):
        with StageTimer(logger, "chroma_snapshot", memory=memory_id) as timer:
            old = _snapshot(collection, memory_id)
            timer.set(records=len(old["ids"]))
        current_dimension = _collection_dimension(collection)
        if current_dimension is not None and current_dimension != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: collection has {current_dimension}, generated {dimension}."
            )
        try:
            upsert_started = time.perf_counter()
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                batch_started = time.perf_counter()
                collection.upsert(ids=ids[start:end], embeddings=vectors[start:end],
                                  documents=texts[start:end], metadatas=metadatas[start:end])
                log_event(logger, "INGEST_CHROMA_BATCH_SUCCESS", memory=memory_id,
                          batch=start // batch_size + 1, start=start,
                          items=len(ids[start:end]), seconds=time.perf_counter() - batch_started)
            log_event(logger, "INGEST_TIMING", stage="chroma_upsert", memory=memory_id,
                      seconds=time.perf_counter() - upsert_started, chunks=len(ids),
                      batches=(len(ids) + batch_size - 1) // batch_size)
            with StageTimer(logger, "chroma_verification", memory=memory_id, chunks=len(ids)):
                _verify_records(collection, memory_id, ids, texts, metadatas, dimension)
            expected_ids = set(ids)
            with StageTimer(logger, "chroma_stale_cleanup", memory=memory_id) as timer:
                stale = [x for x in old["ids"] if x not in expected_ids]
                if stale:
                    collection.delete(ids=stale)
                timer.set(stale_records=len(stale))
            with StageTimer(logger, "chroma_final_verification", memory=memory_id,
                            expected_records=len(expected_ids)) as timer:
                final = collection.get(where={"memory_id": memory_id}, include=["metadatas"])
                final_ids = list(final.get("ids") or [])
                timer.set(actual_records=len(final_ids))
            if set(final_ids) != expected_ids or len(final_ids) != len(expected_ids):
                raise RuntimeError("Chroma verification failed: stale or missing memory chunks remain.")
            log_event(logger, "INGEST_TIMING", stage="chroma_indexing", memory=memory_id,
                      seconds=time.perf_counter() - chroma_started, chunks=len(ids))
            return len(ids)
        except Exception as error:
            log_event(logger, "INGEST_CHROMA_INDEX_ERROR", logging.ERROR, memory=memory_id,
                      error=type(error).__name__, chunks=len(ids),
                      seconds=time.perf_counter() - chroma_started)
            try:
                with StageTimer(logger, "chroma_rollback", memory=memory_id,
                                attempted_records=len(ids), restored_records=len(old["ids"])):
                    _rollback(collection, memory_id, old, ids)
            except Exception as rollback_error:
                log_event(logger, "INGEST_CHROMA_ROLLBACK_ERROR", logging.CRITICAL,
                          memory=memory_id, error=type(rollback_error).__name__)
                raise RuntimeError(
                    "Indexing failed and rollback also failed. Inspect/restore Chroma before retrying. "
                    f"Indexing error: {error}; rollback error: {rollback_error}"
                ) from error
            log_event(logger, "INGEST_CHROMA_ROLLBACK_SUCCESS", logging.WARNING,
                      memory=memory_id, restored_records=len(old["ids"]))
            raise

def verify_memory_index(memory_id: str, expected_chunks: list[Document] | None = None) -> dict[str, Any]:
    """Return a diagnostic report for indexed records belonging to one memory."""
    if not memory_id or not memory_id.strip():
        raise ValueError("Memory ID cannot be empty.")
    with timed_lock(logger, _CHROMA_LOCK, "chroma", "verify_memory_index",
                    memory=memory_id):
        data = get_vector_store()._collection.get(where={"memory_id": memory_id},
                                                   include=["embeddings", "documents", "metadatas"])
    ids = list(data.get("ids") or [])
    docs = list(data.get("documents") or [])
    metas = list(data.get("metadatas") or [])
    vectors = _as_list(data.get("embeddings"))
    errors = []
    if not ids:
        errors.append("No indexed records found for this memory.")
    if not (len(ids) == len(docs) == len(metas) == len(vectors)):
        errors.append("Indexed record fields are incomplete.")
    if len(ids) != len(set(ids)):
        errors.append("Duplicate record IDs were returned.")
    dimensions = {len(_as_list(v)) for v in vectors if _as_list(v)}
    if len(dimensions) > 1:
        errors.append("Indexed vectors have inconsistent dimensions.")
    if any((m or {}).get("memory_id") != memory_id for m in metas):
        errors.append("A record has an incorrect memory_id.")
    if expected_chunks is not None:
        expected_texts = [c.page_content for c in expected_chunks]
        actual = dict(zip(ids, docs))
        expected_ids = [f"{memory_id}_chunk_{i}" for i in range(len(expected_texts))]
        if set(actual) != set(expected_ids):
            errors.append("Indexed IDs do not match expected chunk IDs.")
        for record_id, text in zip(expected_ids, expected_texts):
            if actual.get(record_id) != text:
                errors.append(f"Indexed text mismatch for {record_id}.")
    return {"valid": not errors, "memory_id": memory_id, "record_count": len(ids),
            "embedding_dimension": next(iter(dimensions)) if len(dimensions) == 1 else None,
            "errors": errors, "ids": ids}


class VectorReuseError(Exception):
    """The source memory's vectors cannot be safely reused (``reason`` says why)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def clone_memory_index(*, source_memory_id: str, target_memory_id: str, user_id: str,
                       memory_type: str, expected_texts: list[str]) -> int:
    """Copy an existing memory's vectors to a new memory without re-embedding.

    Safe by construction:
      * the source records must be complete, belong to ``user_id``, have
        contiguous chunk IDs, and their stored text must equal ``expected_texts``
        (the chunks the *current* chunking settings produce from the source's
        extracted text) - so a changed CHUNK_SIZE/OVERLAP or a damaged index is
        detected and reuse is refused;
      * vectors must be finite and match the collection's embedding dimension;
      * new records use the normal ID scheme ``<target>_chunk_<i>`` and metadata
        is copied with ``memory_id``/``user_id``/``memory_type``/``chunk_index``
        updated to the target;
      * the write is verified and rolled back on failure, exactly like
        ``index_memory``. Source records are only ever read.

    Raises ``VectorReuseError`` for any validation failure (caller falls back to
    the normal pipeline). Returns the number of cloned records.
    """
    for name, value in (("source_memory_id", source_memory_id), ("target_memory_id", target_memory_id),
                        ("user_id", user_id), ("memory_type", memory_type)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} cannot be empty.")
    if source_memory_id == target_memory_id:
        raise VectorReuseError("source_is_target")
    if not expected_texts:
        raise VectorReuseError("no_expected_chunks")
    batch_size = int(getattr(settings, "EMBEDDING_BATCH_SIZE", 32))
    if batch_size <= 0:
        raise ValueError("EMBEDDING_BATCH_SIZE must be greater than zero.")

    collection = get_vector_store()._collection
    with timed_lock(logger, _CHROMA_LOCK, "chroma", "read_duplicate_source_vectors",
                    memory=source_memory_id, target_memory=target_memory_id):
        data = collection.get(where={"memory_id": source_memory_id},
                              include=["embeddings", "documents", "metadatas"])
        ids = list(data.get("ids") or [])
        docs = list(data.get("documents") or [])
        metas = list(data.get("metadatas") or [])
        raw_vectors = _as_list(data.get("embeddings"))
        if not ids:
            raise VectorReuseError("source_vectors_missing")
        if not (len(ids) == len(docs) == len(metas) == len(raw_vectors)):
            raise VectorReuseError("source_vectors_incomplete")
        if len(ids) != len(set(ids)):
            raise VectorReuseError("source_duplicate_ids")
        if len(ids) != len(expected_texts):
            raise VectorReuseError("chunk_count_mismatch")

        position = {record_id: i for i, record_id in enumerate(ids)}
        new_ids, new_vectors, new_metadatas = [], [], []
        dimension: int | None = None
        for index, expected_text in enumerate(expected_texts):
            pos = position.get(f"{source_memory_id}_chunk_{index}")
            if pos is None:
                raise VectorReuseError("chunk_ids_mismatch")
            if docs[pos] != expected_text:
                raise VectorReuseError("chunk_text_mismatch")
            meta = dict(metas[pos] or {})
            if meta.get("memory_id") != source_memory_id or meta.get("user_id") != user_id:
                raise VectorReuseError("source_metadata_mismatch")
            vector = [float(x) for x in _as_list(raw_vectors[pos])]
            if not vector or any(not math.isfinite(x) for x in vector):
                raise VectorReuseError("source_vector_invalid")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise VectorReuseError("dimension_inconsistent")
            meta.update({"memory_id": target_memory_id, "user_id": user_id,
                         "memory_type": memory_type, "chunk_index": index})
            new_ids.append(f"{target_memory_id}_chunk_{index}")
            new_vectors.append(vector)
            new_metadatas.append(_safe_metadata(meta))

        collection_dimension = _collection_dimension(collection)
        if collection_dimension is not None and collection_dimension != dimension:
            raise VectorReuseError("dimension_mismatch")

        with StageTimer(logger, "chroma_snapshot", memory=target_memory_id) as timer:
            old = _snapshot(collection, target_memory_id)
            timer.set(records=len(old["ids"]))
        try:
            upsert_started = time.perf_counter()
            for start in range(0, len(new_ids), batch_size):
                end = start + batch_size
                collection.upsert(ids=new_ids[start:end], embeddings=new_vectors[start:end],
                                  documents=list(expected_texts[start:end]),
                                  metadatas=new_metadatas[start:end])
            log_event(logger, "INGEST_TIMING", stage="chroma_clone_upsert", memory=target_memory_id,
                      seconds=time.perf_counter() - upsert_started, chunks=len(new_ids),
                      batches=(len(new_ids) + batch_size - 1) // batch_size)
            with StageTimer(logger, "chroma_verification", memory=target_memory_id,
                            chunks=len(new_ids)):
                _verify_records(collection, target_memory_id, new_ids, list(expected_texts),
                                new_metadatas, dimension)
            expected_set = set(new_ids)
            with StageTimer(logger, "chroma_stale_cleanup", memory=target_memory_id) as timer:
                stale = [x for x in old["ids"] if x not in expected_set]
                if stale:
                    collection.delete(ids=stale)
                timer.set(stale_records=len(stale))
            with StageTimer(logger, "chroma_final_verification", memory=target_memory_id,
                            expected_records=len(expected_set)) as timer:
                final = collection.get(where={"memory_id": target_memory_id}, include=["metadatas"])
                final_ids = list(final.get("ids") or [])
                timer.set(actual_records=len(final_ids))
            if set(final_ids) != expected_set or len(final_ids) != len(expected_set):
                raise RuntimeError("Chroma verification failed: stale or missing memory chunks remain.")
            return len(new_ids)
        except Exception as error:
            try:
                with StageTimer(logger, "chroma_rollback", memory=target_memory_id,
                                attempted_records=len(new_ids), restored_records=len(old["ids"])):
                    _rollback(collection, target_memory_id, old, new_ids)
            except Exception as rollback_error:
                raise RuntimeError(
                    "Vector clone failed and rollback also failed. Inspect/restore Chroma before retrying. "
                    f"Clone error: {type(error).__name__}; rollback error: {type(rollback_error).__name__}"
                ) from error
            raise
