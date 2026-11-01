DEFAULT_DISTANCE_THRESHOLD = 1.2


def get_grounded_results(
    results,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
):
    """Keep chunks that are semantically close OR contain the query's keywords.

    ``distance`` is the dense (semantic) distance. ``keyword_match`` is set by
    rag/retriever.py when the chunk contains enough of the query's content
    words, so exact names, dates and identifiers are not dropped just because
    their embedding is far away. Hybrid (RRF) scores are rank-based and are
    never compared with the distance threshold.
    """

    grounded_results = [
        (document, distance)
        for document, distance in results
        if distance <= threshold
        or document.metadata.get("keyword_match", False)
    ]

    return grounded_results


def is_grounded(
    results,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
):

    return bool(
        get_grounded_results(
            results,
            threshold
        )
    )