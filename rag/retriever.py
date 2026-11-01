import math

from langchain_core.documents import Document
from langsmith import traceable
from qdrant_client import models

from config.settings import settings
from rag.embeddings import get_embeddings
from rag.vector_store import (
    COLLECTION_NAME,
    DENSE_VECTOR,
    QDRANT_LOCK,
    SPARSE_VECTOR,
    TEXT_KEY,
    get_client,
    query_terms,
    sparse_query,
)


DEFAULT_TOP_K = 3

# Each search branch (dense / BM25) contributes this many candidates to RRF.
CANDIDATES_PER_BRANCH = 20

# A chunk counts as a keyword match when it contains at least this share of the
# query's content words (stop words excluded).
KEYWORD_MIN_COVERAGE = 0.5

# Largest possible value of 2 - 2*cosine; used if a chunk has no stored vector.
MAX_DISTANCE = 4.0


def _build_filter(user_id: str, memory_id: str | None) -> models.Filter:
    """Always scope by user; optionally by one memory."""
    must = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ]
    if memory_id is not None:
        must.append(
            models.FieldCondition(key="memory_id", match=models.MatchValue(value=memory_id))
        )
    return models.Filter(must=must)


def _dense_distance(query_vector: list[float], chunk_vector: list[float] | None) -> float:
    """Distance on the same scale the app has always used (0 = identical).

    ``2 - 2 * cosine`` equals the squared-L2 distance of unit vectors, which is
    what Chroma returned for this embedding model. This keeps the grounding
    threshold (1.2) and confidence cut-off (0.4) meaningful.
    RRF scores are rank-based and are NEVER used as distances.
    """
    if not chunk_vector:
        return MAX_DISTANCE
    dot = sum(a * b for a, b in zip(query_vector, chunk_vector))
    norm = math.sqrt(sum(a * a for a in query_vector)) * math.sqrt(sum(b * b for b in chunk_vector))
    if norm == 0:
        return MAX_DISTANCE
    return max(0.0, 2.0 - 2.0 * dot / norm)


def _search(client, mode: str, dense, sparse, flt, top_k: int):
    common = dict(
        collection_name=COLLECTION_NAME,
        query_filter=flt,
        limit=top_k,
        with_payload=True,
        with_vectors=[DENSE_VECTOR],
    )

    if mode == "bm25":
        if sparse is None:
            return []
        return client.query_points(query=sparse, using=SPARSE_VECTOR, **common).points

    # Nothing to match by keyword (e.g. only stop words): dense search only.
    if mode == "dense" or sparse is None:
        return client.query_points(query=dense, using=DENSE_VECTOR, **common).points

    # Hybrid: both branches carry the same user/memory filter; Qdrant fuses
    # the two rankings with Reciprocal Rank Fusion.
    return client.query_points(
        prefetch=[
            models.Prefetch(query=dense, using=DENSE_VECTOR, filter=flt,
                            limit=CANDIDATES_PER_BRANCH),
            models.Prefetch(query=sparse, using=SPARSE_VECTOR, filter=flt,
                            limit=CANDIDATES_PER_BRANCH),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        **common,
    ).points


@traceable(name="Memory Retriever")
def retrieve(
    question: str,
    user_id: str,
    top_k: int = DEFAULT_TOP_K,
    memory_id: str | None = None,
    mode: str | None = None,
):
    """Return ``[(Document, distance), ...]`` ranked best first.

    ``distance`` is the dense (semantic) distance, lower is better. Each
    Document's metadata also carries ``keyword_match`` (bool), which
    rag/grounding.py uses so exact-term hits are not discarded.
    ``mode`` overrides settings.RETRIEVAL_MODE ("hybrid", "dense", "bm25").
    """
    # ============================================================
    # Validate input
    # ============================================================

    if not question or not question.strip():
        return []

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID is required for memory retrieval."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if memory_id is not None and not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    mode = (mode or settings.RETRIEVAL_MODE).lower()
    if mode not in {"hybrid", "dense", "bm25"}:
        raise ValueError("mode must be one of: hybrid, dense, bm25.")

    # ============================================================
    # Search (global: user filter only; memory-specific: user + memory)
    # ============================================================

    question = question.strip()
    dense_vector = get_embeddings().embed_query(question)
    sparse_vector = sparse_query(question)
    flt = _build_filter(user_id, memory_id)

    client = get_client()
    with QDRANT_LOCK:
        points = _search(client, mode, dense_vector, sparse_vector, flt, top_k)

    # ============================================================
    # Convert to (Document, distance)
    # ============================================================

    terms = set(query_terms(question))
    results = []
    for point in points:
        payload = dict(point.payload or {})
        text = payload.pop(TEXT_KEY, "")
        if not text:
            continue

        coverage = (
            len(terms & set(query_terms(text))) / len(terms) if terms else 0.0
        )
        payload["keyword_match"] = coverage >= KEYWORD_MIN_COVERAGE

        chunk_vector = (point.vector or {}).get(DENSE_VECTOR)
        distance = _dense_distance(dense_vector, chunk_vector)

        results.append((Document(page_content=text, metadata=payload), distance))

    return results