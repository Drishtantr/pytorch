"""K-fold cross-validation helpers for the from-scratch chapters.

A single held-out split makes every quality number depend on one arbitrary
partition. These helpers score a model on ``k`` disjoint validation folds so a
chapter can report a mean and a spread instead.

Plain Python lists and the standard library only -- no numpy or torch.
"""

import random
import statistics
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple


class CrossValidationResult(NamedTuple):
    """Per-fold scores plus their mean and (population) standard deviation."""

    scores: list[float]
    mean: float
    std: float


def k_folds(
    n_samples: int, k: int, shuffle: bool = True, seed: int | None = None
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield ``k`` ``(train_indices, val_indices)`` pairs over ``range(n_samples)``.

    Every index lands in exactly one validation fold, and within a fold the
    train and validation indices never overlap. When ``n_samples`` does not
    divide evenly by ``k``, the remainder goes one apiece to the earliest
    folds, so no two fold sizes differ by more than one.

        >>> [val for _, val in k_folds(5, 2, shuffle=False)]
        [[0, 1, 2], [3, 4]]

    With ``shuffle=True`` the indices are permuted first; the same ``seed``
    gives the same folds, and ``seed=None`` gives a fresh permutation each call.

    Raises ``ValueError`` when ``k`` is below 2 or above ``n_samples``. The
    check runs at call time, not on first iteration.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    if k > n_samples:
        raise ValueError(f"k ({k}) cannot exceed n_samples ({n_samples})")

    indices = list(range(n_samples))
    if shuffle:
        random.Random(seed).shuffle(indices)

    return _iter_folds(indices, k)


def _iter_folds(indices: list[int], k: int) -> Iterator[tuple[list[int], list[int]]]:
    base, remainder = divmod(len(indices), k)
    start = 0
    for fold in range(k):
        size = base + 1 if fold < remainder else base
        stop = start + size
        yield indices[:start] + indices[stop:], indices[start:stop]
        start = stop


def cross_validate(
    features: Sequence[Any],
    targets: Sequence[Any],
    fit_predict: Callable[[list[Any], list[Any], list[Any], list[Any]], float],
    k: int = 5,
    *,
    shuffle: bool = True,
    seed: int | None = None,
) -> CrossValidationResult:
    """Score ``fit_predict`` on each of ``k`` folds and summarise the scores.

    ``fit_predict(train_x, train_y, val_x, val_y)`` is called once per fold. It
    should fit a fresh model on the training rows, predict the validation rows,
    and return a single score for that fold (for example, validation MSE).

    Returns a ``CrossValidationResult`` of the per-fold scores in fold order,
    their mean, and their population standard deviation (``statistics.pstdev``).
    Pass ``seed`` to make the folds, and so the scores, reproducible.

    Raises ``ValueError`` when ``features`` and ``targets`` differ in length,
    or when ``k`` is below 2 or above the number of samples.
    """
    if len(features) != len(targets):
        raise ValueError(
            f"features ({len(features)}) and targets ({len(targets)}) "
            "must be the same length"
        )

    scores = []
    for train_idx, val_idx in k_folds(len(features), k, shuffle=shuffle, seed=seed):
        score = fit_predict(
            [features[i] for i in train_idx],
            [targets[i] for i in train_idx],
            [features[i] for i in val_idx],
            [targets[i] for i in val_idx],
        )
        scores.append(float(score))

    return CrossValidationResult(
        scores=scores,
        mean=statistics.fmean(scores),
        std=statistics.pstdev(scores),
    )
