"""Dependency-free data helpers for the from-scratch chapters.

The early notebooks build regression by hand, so they cannot lean on
``torch.utils.data`` yet. These helpers keep the cells short: shuffling a
dataset into train/validation splits, walking it in minibatches, scaling
features, and tracking a running loss across an epoch.

Everything operates on plain Python lists of floats so the notebooks stay
readable before tensors are introduced.

Randomness flows through a single module-level generator. Call :func:`set_seed`
once and both the split and the batch order become reproducible; nothing here
touches the global :mod:`random` state, so seeding elsewhere in a notebook is
left alone.
"""

import math
import random
from collections.abc import Iterable, Iterator, Sequence

DEFAULT_BATCH_SIZE = 32
DEFAULT_VAL_FRACTION = 0.2

# Every shuffle in this module draws from here so one seed covers all of them.
_rng = random.Random()

# A feature whose values are all identical carries no signal; scaling it would
# divide by zero, so it is centred to zeros instead.
_ZERO_VARIANCE_STD = 0.0


def set_seed(seed: int) -> None:
    """Seed the module generator so splits and batch order are reproducible."""
    _rng.seed(seed)


def train_val_split(
    features: Sequence,
    targets: Sequence,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    rng: random.Random | None = None,
) -> tuple[list, list, list, list]:
    """Shuffle the dataset and cut it into train and validation splits.

    Returns ``(train_x, train_y, val_x, val_y)``. A single permutation is
    applied to both sequences, so every feature row keeps its own target.

    Pass ``rng`` to draw from a specific generator; otherwise the module
    generator is used, which :func:`set_seed` controls.
    """
    if len(features) != len(targets):
        raise ValueError(
            f"features and targets must be the same length, "
            f"got {len(features)} and {len(targets)}"
        )
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0.0, 1.0), got {val_fraction}")

    generator = _rng if rng is None else rng

    # Shuffle an index permutation, not the two lists separately -- shuffling
    # each in turn would pair every row with some other row's target.
    order = list(range(len(features)))
    generator.shuffle(order)

    n_val = int(len(features) * val_fraction)
    val_idx, train_idx = order[:n_val], order[n_val:]

    return (
        [features[i] for i in train_idx],
        [targets[i] for i in train_idx],
        [features[i] for i in val_idx],
        [targets[i] for i in val_idx],
    )


def num_batches(
    data: Sequence,
    batch_size: int = DEFAULT_BATCH_SIZE,
    drop_last: bool = False,
) -> int:
    """Number of minibatches :func:`iter_batches` yields for ``data``.

    Mirrors :func:`iter_batches` exactly, including the ``drop_last`` handling,
    so a progress bar or LR schedule driven off this count stays in step with
    the loop.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if drop_last:
        return len(data) // batch_size
    return math.ceil(len(data) / batch_size)


def iter_batches(
    features: Sequence,
    targets: Sequence,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    drop_last: bool = False,
    rng: random.Random | None = None,
) -> Iterator[tuple[list, list]]:
    """Yield ``(batch_x, batch_y)`` pairs walking the dataset once.

    Every row is visited exactly once. The final batch is short when the
    dataset does not divide evenly; set ``drop_last`` to skip it instead, which
    is what you want if a downstream step misbehaves on a batch of one.

    When ``shuffle`` is set the row order is randomised each epoch, drawing
    from the module generator unless ``rng`` is given.
    """
    if len(features) != len(targets):
        raise ValueError(
            f"features and targets must be the same length, "
            f"got {len(features)} and {len(targets)}"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    generator = _rng if rng is None else rng

    order = list(range(len(features)))
    if shuffle:
        generator.shuffle(order)

    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        if drop_last and len(rows) < batch_size:
            break
        yield [features[i] for i in rows], [targets[i] for i in rows]


def fit_standardizer(values: Iterable[float]) -> tuple[float, float]:
    """Return the ``(mean, std)`` of ``values`` for later rescaling.

    Fit this on the training split alone and replay it on validation with
    :func:`apply_standardizer`; fitting on the full dataset would leak
    validation statistics into training.

    ``std`` is the population standard deviation (divides by ``n``), matching
    the convention used by the usual feature scalers.
    """
    values = list(values)
    if not values:
        raise ValueError("cannot fit a standardizer on an empty sequence")

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(variance)


def apply_standardizer(
    values: Iterable[float],
    stats: tuple[float, float],
) -> list[float]:
    """Return a new list holding ``values`` rescaled by ``stats``.

    The input is left untouched. A zero standard deviation means the fitted
    feature was constant, so values are centred and left at zero rather than
    raising.
    """
    mean, std = stats
    if std == _ZERO_VARIANCE_STD:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def standardize(values: Iterable[float]) -> tuple[list[float], tuple[float, float]]:
    """Rescale ``values`` to zero mean and unit variance.

    Returns ``(scaled, stats)``, where ``stats`` can be handed to
    :func:`apply_standardizer` to put another split on the same scale. The
    input sequence is not modified.
    """
    values = list(values)
    stats = fit_standardizer(values)
    return apply_standardizer(values, stats), stats


class AverageMeter:
    """Track a running mean, e.g. the loss over one epoch."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Fold ``value``, observed over ``n`` samples, into the average.

        ``value`` is treated as a per-sample mean, so it is weighted by ``n``.
        That keeps a short final batch from counting as much as a full one.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        """Mean of everything folded in so far, or 0.0 when empty."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
