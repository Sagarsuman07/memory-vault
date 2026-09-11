from langsmith import traceable

from rag.vector_store import get_vector_store


DEFAULT_TOP_K = 3


@traceable(name="Memory Retriever")
def retrieve(
    question: str,
    user_id: str,
    top_k: int = DEFAULT_TOP_K,
    memory_id: str | None = None,
):
    # ============================================================
    # Validate question
    # ============================================================

    if not question or not question.strip():
        return []

    # ============================================================
    # Validate user
    # ============================================================

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID is required for memory retrieval."
        )

    # ============================================================
    # Validate top_k
    # ============================================================

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    # ============================================================
    # Get vector store
    # ============================================================

    vector_store = get_vector_store()

    # ============================================================
    # Global Search
    #
    # Search across all memories belonging to
    # the current user.
    # ============================================================

    if memory_id is None:

        filters = {
            "user_id": user_id
        }

    # ============================================================
    # Memory-Specific Search
    #
    # Search only inside the selected memory AND
    # make sure it belongs to the current user.
    # ============================================================

    else:

        if not memory_id.strip():
            raise ValueError(
                "Memory ID cannot be empty."
            )

        filters = {
            "$and": [
                {
                    "user_id": user_id
                },
                {
                    "memory_id": memory_id
                },
            ]
        }

    # ============================================================
    # Semantic Searchs
    # ============================================================

    results = vector_store.similarity_search_with_score(
        question,
        k=top_k,
        filter=filters,
    )

    return results