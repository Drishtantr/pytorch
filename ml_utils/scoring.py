"""Scoring helpers shared across the practice repos."""


def pass_rate(passed: int, total: int) -> float:
    """Share of cases that passed, as a **percentage from 0 to 100**.

    Note the unit: this returns ``95.0`` for 19 of 20, not ``0.95``. Callers
    comparing the result against a threshold need that threshold on the same
    0-100 scale.

        >>> pass_rate(19, 20)
        95.0
        >>> pass_rate(0, 4)
        0.0
        >>> pass_rate(4, 4)
        100.0

    Raises ``ValueError`` when ``total`` is not positive, when ``passed`` is
    negative, or when ``passed`` exceeds ``total``.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if passed < 0:
        raise ValueError(f"passed must be non-negative, got {passed}")
    if passed > total:
        raise ValueError(f"passed ({passed}) cannot exceed total ({total})")

    return 100.0 * passed / total
