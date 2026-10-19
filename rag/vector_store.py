"""Safe, batched, verifiable Chroma indexing for Memory Vault."""
from __future__ import annotations
import math
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import settings
from rag.embeddings import get_embeddings

CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "memory_vault"


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
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        raw = embedder.embed_documents(batch)
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
    """Batch-embed and safely replace one memory's deterministic Chroma records."""
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
        # Application-controlled identity fields always win.
        metadata.update({"memory_id": memory_id, "user_id": user_id,
                         "memory_type": memory_type, "chunk_index": i})
        documents.append(Document(page_content=text, metadata=metadata))
        # Preserve the established ID format for compatibility.
        ids.append(f"{memory_id}_chunk_{i}")

    texts = [d.page_content for d in documents]
    metadatas = [dict(d.metadata) for d in documents]
    batch_size = int(getattr(settings, "EMBEDDING_BATCH_SIZE", 32))
    if batch_size <= 0:
        raise ValueError("EMBEDDING_BATCH_SIZE must be greater than zero.")

    collection = get_vector_store()._collection
    old = _snapshot(collection, memory_id)
    existing_dimension = _collection_dimension(collection)
    try:
        if progress_callback:
            progress_callback("embedding")
        vectors = _embed_batches(texts, get_embeddings(), batch_size, existing_dimension)
        dimension = len(vectors[0])
        if progress_callback:
            progress_callback("indexing")
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(ids=ids[start:end], embeddings=vectors[start:end],
                              documents=texts[start:end], metadatas=metadatas[start:end])
        _verify_records(collection, memory_id, ids, texts, metadatas, dimension)
        stale = [x for x in old["ids"] if x not in set(ids)]
        if stale:
            collection.delete(ids=stale)
        final = collection.get(where={"memory_id": memory_id}, include=["metadatas"])
        final_ids = list(final.get("ids") or [])
        if set(final_ids) != set(ids) or len(final_ids) != len(ids):
            raise RuntimeError("Chroma verification failed: stale or missing memory chunks remain.")
        return len(ids)
    except Exception as error:
        try:
            _rollback(collection, memory_id, old, ids)
        except Exception as rollback_error:
            raise RuntimeError("Indexing failed and rollback also failed. Inspect/restore Chroma before retrying. "
                               f"Indexing error: {error}; rollback error: {rollback_error}") from error
        raise


def verify_memory_index(memory_id: str, expected_chunks: list[Document] | None = None) -> dict[str, Any]:
    """Return a diagnostic report for indexed records belonging to one memory."""
    if not memory_id or not memory_id.strip():
        raise ValueError("Memory ID cannot be empty.")
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
