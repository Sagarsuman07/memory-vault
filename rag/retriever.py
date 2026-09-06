from rag.vector_store import get_vector_store


DEFAULT_TOP_K = 3


def retrieve(
    question: str,
    user_id: str,
    top_k: int = DEFAULT_TOP_K,
    memory_id: str | None = None,
):

    if not question or not question.strip():

        return []


    vector_store = get_vector_store()


    # ========================================================
    # Global Search
    # ========================================================
    #
    # Search all memories belonging to the current user.
    #
    # Example:
    #
    # {
    #     "user_id": "demo_user"
    # }
    #
    # ========================================================

    if memory_id is None:

        filters = {
            "user_id": user_id
        }


    # ========================================================
    # Memory-Specific Search
    # ========================================================
    #
    # Search only the selected memory belonging to
    # the current user.
    #
    # Chroma requires multiple conditions to be combined
    # using $and.
    #
    # ========================================================

    else:

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


    # ========================================================
    # Semantic Search
    # ========================================================

    results = vector_store.similarity_search_with_score(
        question,
        k=top_k,
        filter=filters,
    )


    return results