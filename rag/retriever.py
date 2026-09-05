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

    filters = {
        "user_id": user_id
    }

    if memory_id is not None:

        filters["memory_id"] = memory_id

    results = vector_store.similarity_search_with_score(
        question,
        k=top_k,
        filter=filters,
    )

    return results