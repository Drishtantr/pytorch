"""Ranking helpers shared across the practice repos."""


def top_k(scores: list[float], k: int) -> list[tuple[int, float]]:
    """The ``k`` highest scores as ``(index, score)`` pairs, **best first**.

    Ordered by score descending, so the highest score is at position 0 and
    ``top_k(scores, k)[0]`` is the winner.

        >>> top_k([0.2, 0.9, 0.5], 2)
        [(1, 0.9), (2, 0.5)]

    Raises ``ValueError`` when ``k`` is not positive or exceeds ``len(scores)``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(scores):
        raise ValueError(f"k ({k}) cannot exceed len(scores) ({len(scores)})")

    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]
