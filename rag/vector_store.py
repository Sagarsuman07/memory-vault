"""Safe, batched, verifiable Chroma indexing for Memory Vault."""
from __future__ import annotations
import math
import logging
import time
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import threading
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import settings
from config.logging_config import configure_logging
from rag.embeddings import get_embeddings

configure_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "memory_vault"

# One lock protects the shared Chroma collection across worker threads. Keep it
# process-wide; a lock created per job would not coordinate concurrent writes.
_CHROMA_LOCK = threading.RLock()
_EMBEDDING_LOCK = threading.Semaphore(
    max(1, int(getattr(settings, "EMBEDDING_CONCURRENCY", 1)))
)


def get_vector_store():
    """Return the existing persistent Chroma vector store."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(collection_name=COLLECTION_NAME,
                  embedding_function=get_embeddings(),
                  persist_directory=str(CHROMA_DIR))


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
    total_embedding_seconds = 0.0
    batch_count = 0
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        # Release the shared embedding slot after each batch, so a small memory
        # can make progress between batches of a large memory.
        batch_started = time.perf_counter()
        with _EMBEDDING_LOCK:
            raw = embedder.embed_documents(batch)
        batch_seconds = time.perf_counter() - batch_started
        total_embedding_seconds += batch_seconds
        batch_count += 1
        logger.info("INGEST_TIMING stage=embedding_batch batch=%d batch_chunks=%d total_chunks=%d seconds=%.3f", batch_count, len(batch), len(texts), batch_seconds)
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
    logger.info("INGEST_TIMING stage=embedding_total seconds=%.3f chunks=%d batches=%d dimension=%s", total_embedding_seconds, len(texts), batch_count, dimension)
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
    dimension_check_started = time.perf_counter()
    with _CHROMA_LOCK:
        existing_dimension = _collection_dimension(collection)
    logger.info("INGEST_TIMING stage=chroma_dimension_check seconds=%.3f dimension=%s", time.perf_counter() - dimension_check_started, existing_dimension)

    if progress_callback:
        progress_callback("embedding")
    # The supplied rag/embeddings.py uses HuggingFaceEmbeddings locally, not an
    # embedding API. _embed_batches limits local inference per batch to reduce
    # CPU/RAM spikes while allowing other jobs between batches.
    vectors = _embed_batches(texts, get_embeddings(), batch_size, existing_dimension)
    dimension = len(vectors[0])

    if progress_callback:
        progress_callback("indexing")
    # Snapshot + upsert + verify + stale cleanup + rollback are one critical
    # section so two jobs cannot interleave writes to the shared collection.
    transaction_started = time.perf_counter()
    with _CHROMA_LOCK:
        lock_acquired_at = time.perf_counter()
        logger.info("INGEST_TIMING stage=chroma_lock_wait memory_id=%s chunks=%d seconds=%.3f", memory_id, len(ids), lock_acquired_at - transaction_started)
        snapshot_started = time.perf_counter()
        old = _snapshot(collection, memory_id)
        snapshot_seconds = time.perf_counter() - snapshot_started
        current_dimension = _collection_dimension(collection)
        if current_dimension is not None and current_dimension != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: collection has {current_dimension}, generated {dimension}."
            )
        try:
            write_started = time.perf_counter()
            write_batches = 0
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                collection.upsert(ids=ids[start:end], embeddings=vectors[start:end],
                                  documents=texts[start:end], metadatas=metadatas[start:end])
                write_batches += 1
            write_seconds = time.perf_counter() - write_started
            verify_started = time.perf_counter()
            _verify_records(collection, memory_id, ids, texts, metadatas, dimension)
            verify_seconds = time.perf_counter() - verify_started
            cleanup_started = time.perf_counter()
            expected_ids = set(ids)
            stale = [x for x in old["ids"] if x not in expected_ids]
            if stale:
                collection.delete(ids=stale)
            final = collection.get(where={"memory_id": memory_id}, include=["metadatas"])
            final_ids = list(final.get("ids") or [])
            if set(final_ids) != expected_ids or len(final_ids) != len(expected_ids):
                raise RuntimeError("Chroma verification failed: stale or missing memory chunks remain.")
            cleanup_seconds = time.perf_counter() - cleanup_started
            logger.info("INGEST_TIMING stage=chroma_indexing memory_id=%s chunks=%d batches=%d snapshot_seconds=%.3f write_seconds=%.3f verify_seconds=%.3f cleanup_seconds=%.3f lock_held_seconds=%.3f", memory_id, len(ids), write_batches, snapshot_seconds, write_seconds, verify_seconds, cleanup_seconds, time.perf_counter() - lock_acquired_at)
            return len(ids)
        except Exception as error:
            logger.exception("INGEST_TIMING stage=chroma_indexing_failed memory_id=%s chunks=%d elapsed_seconds=%.3f", memory_id, len(ids), time.perf_counter() - lock_acquired_at)
            try:
                rollback_started = time.perf_counter()
                _rollback(collection, memory_id, old, ids)
                logger.info("INGEST_TIMING stage=chroma_rollback memory_id=%s seconds=%.3f", memory_id, time.perf_counter() - rollback_started)
            except Exception as rollback_error:
                raise RuntimeError(
                    "Indexing failed and rollback also failed. Inspect/restore Chroma before retrying. "
                    f"Indexing error: {error}; rollback error: {rollback_error}"
                ) from error
            raise

def get_memory_index_text(memory_id: str, user_id: str) -> str:
    """Reconstruct stored text from this user's existing indexed chunks.

    Used only to recover missing legacy SQL extracted_text during duplicate reuse.
    It never calls the embedding model or modifies Chroma.
    """
    if not isinstance(memory_id, str) or not memory_id.strip():
        raise ValueError("memory_id cannot be empty.")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id cannot be empty.")
    collection = get_vector_store()._collection
    with _CHROMA_LOCK:
        data = collection.get(
            where={"memory_id": memory_id},
            include=["documents", "metadatas"],
        )
    documents = list(data.get("documents") or [])
    metadatas = list(data.get("metadatas") or [])
    if not documents or len(documents) != len(metadatas):
        return ""
    records: list[tuple[int, str]] = []
    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        if metadata.get("memory_id") != memory_id or metadata.get("user_id") != user_id:
            raise ValueError("Indexed chunks do not belong to the requested user and memory.")
        index = metadata.get("chunk_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Indexed chunk metadata has an invalid chunk_index.")
        if not isinstance(document, str) or not document.strip():
            raise ValueError("Indexed chunk text is empty.")
        records.append((index, document))
    records.sort(key=lambda item: item[0])
    indices = [index for index, _ in records]
    if indices != list(range(len(records))):
        raise ValueError("Indexed chunk sequence is incomplete or duplicated.")
    return "\n\n".join(document for _, document in records)


def clone_memory_index(
    source_memory_id: str,
    target_memory_id: str,
    user_id: str,
    memory_type: str,
) -> int:
    """Copy a verified memory's stored vectors to a new memory identity.

    This deliberately reuses existing embeddings and document chunks. It never
    modifies or deletes the source memory's Chroma records. The target write is
    protected by the same lock, verified, and rolled back on failure.
    """
    for name, value in (("source_memory_id", source_memory_id),
                        ("target_memory_id", target_memory_id),
                        ("user_id", user_id), ("memory_type", memory_type)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} cannot be empty.")
    if source_memory_id == target_memory_id:
        raise ValueError("Source and target memory IDs must be different.")

    clone_started = time.perf_counter()
    collection = get_vector_store()._collection
    with _CHROMA_LOCK:
        source = collection.get(
            where={"memory_id": source_memory_id},
            include=["embeddings", "documents", "metadatas"],
        )
        source_ids = list(source.get("ids") or [])
        source_docs = list(source.get("documents") or [])
        source_metas = list(source.get("metadatas") or [])
        source_vectors = _as_list(source.get("embeddings"))

        if not source_ids:
            raise ValueError("The source memory has no indexed vectors to reuse.")
        if not (len(source_ids) == len(source_docs) == len(source_metas) == len(source_vectors)):
            raise RuntimeError("Source vector records are incomplete; reuse was stopped safely.")

        records = []
        seen_indices: set[int] = set()
        dimensions: set[int] = set()
        for record_id, text, raw_meta, raw_vector in zip(
            source_ids, source_docs, source_metas, source_vectors
        ):
            metadata = dict(raw_meta or {})
            if metadata.get("memory_id") != source_memory_id:
                raise ValueError("Source index contains a record with an unexpected memory ID.")
            if metadata.get("user_id") != user_id:
                raise ValueError("Source index does not belong to the requesting user.")
            index = metadata.get("chunk_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen_indices:
                raise ValueError("Source index has missing or duplicate chunk indices.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Source index contains an empty chunk.")
            vector = [float(value) for value in _as_list(raw_vector)]
            if not vector or any(not math.isfinite(value) for value in vector):
                raise ValueError("Source index contains an invalid embedding vector.")
            seen_indices.add(index)
            dimensions.add(len(vector))
            records.append((index, record_id, text, metadata, vector))

        if seen_indices != set(range(len(records))):
            raise ValueError("Source chunk indices are not contiguous; reuse was stopped safely.")
        if len(dimensions) != 1:
            raise ValueError("Source index contains inconsistent embedding dimensions.")
        records.sort(key=lambda record: record[0])

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        for index, _source_id, text, source_metadata, vector in records:
            target_id = f"{target_memory_id}_chunk_{index}"
            metadata = dict(source_metadata)
            metadata.update({
                "memory_id": target_memory_id,
                "user_id": user_id,
                "memory_type": memory_type,
                "chunk_index": index,
            })
            ids.append(target_id)
            texts.append(text)
            metadatas.append(_safe_metadata(metadata))
            vectors.append(vector)

        dimension = len(vectors[0])
        collection_dimension = _collection_dimension(collection)
        if collection_dimension is not None and collection_dimension != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: collection has {collection_dimension}, source has {dimension}."
            )

        old = _snapshot(collection, target_memory_id)
        try:
            batch_size = max(1, int(getattr(settings, "EMBEDDING_BATCH_SIZE", 32)))
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                collection.upsert(
                    ids=ids[start:end],
                    embeddings=vectors[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                )
            _verify_records(collection, target_memory_id, ids, texts, metadatas, dimension)
            current = collection.get(where={"memory_id": target_memory_id}, include=["metadatas"])
            current_ids = list(current.get("ids") or [])
            if set(current_ids) != set(ids) or len(current_ids) != len(ids):
                raise RuntimeError("Cloned index verification found missing or stale target records.")
            stale = [record_id for record_id in old["ids"] if record_id not in set(ids)]
            if stale:
                collection.delete(ids=stale)
            logger.info("INGEST_TIMING stage=clone_memory_index source_memory_id=%s target_memory_id=%s chunks=%d seconds=%.3f", source_memory_id, target_memory_id, len(ids), time.perf_counter() - clone_started)
            return len(ids)
        except Exception as error:
            try:
                _rollback(collection, target_memory_id, old, ids)
            except Exception as rollback_error:
                raise RuntimeError(
                    "Vector reuse failed and rollback also failed. Inspect/restore Chroma before retrying. "
                    f"Reuse error: {error}; rollback error: {rollback_error}"
                ) from error
            raise


def verify_memory_index(memory_id: str, expected_chunks: list[Document] | None = None) -> dict[str, Any]:
    """Return a diagnostic report for indexed records belonging to one memory."""
    if not memory_id or not memory_id.strip():
        raise ValueError("Memory ID cannot be empty.")
    with _CHROMA_LOCK:
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
