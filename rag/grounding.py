DEFAULT_DISTANCE_THRESHOLD = 0.7


def get_grounded_results(
    results,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
):

    grounded_results = [
        (document, distance)
        for document, distance in results
        if distance <= threshold
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