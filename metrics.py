"""Regression metrics for the notebook exercises.

Pure-Python helpers so the notebooks can report error terms without
pulling in an extra dependency.
"""


def mean_squared_error(actual, predicted):
    """Return the mean of the squared differences."""
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must be the same length")
    if not actual:
        raise ValueError("actual must not be empty")

    total = 0.0
    for a, p in zip(actual, predicted):
        total += (a - p) ** 2
    return total / len(actual)


def mean_absolute_error(actual, predicted):
    """Return the mean of the absolute differences."""
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must be the same length")
    if not actual:
        raise ValueError("actual must not be empty")

    total = 0.0
    for a, p in zip(actual, predicted):
        total += abs(a - p)
    return total / len(actual)


def r_squared(actual, predicted):
    """Return the coefficient of determination.

    Returns 0.0 when the observations have no variance, matching the
    convention scikit-learn uses.
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must be the same length")
    if not actual:
        raise ValueError("actual must not be empty")

    mean_actual = sum(actual) / len(actual)

    ss_res = 0.0
    for a, p in zip(actual, predicted):
        ss_res += (a - p) ** 2

    ss_tot = 0.0
    for a in actual:
        ss_tot += (a - mean_actual) ** 2

    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)
