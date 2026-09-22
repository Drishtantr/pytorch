"""Dependency-free data helpers for the from-scratch chapters.

The early notebooks build regression by hand, so they cannot lean on
``torch.utils.data`` yet. These helpers keep the cells short: shuffling a
dataset into train/validation splits, walking it in minibatches, scaling
features, and tracking a running loss across an epoch.

Everything operates on plain Python lists of floats so the notebooks stay
readable before tensors are introduced.
"""

import math
import random

DEFAULT_BATCH_SIZE = 32
DEFAULT_VAL_FRACTION = 0.2


def set_seed(seed: int) -> None:
    """Seed the module RNG so splits and batch order are reproducible."""
    random.seed(seed)


def train_val_split(features, targets, val_fraction=DEFAULT_VAL_FRACTION, seed=0):
    """Shuffle the dataset and cut it into train and validation halves.

    Returns ``(train_x, train_y, val_x, val_y)``. The shuffle keeps each
    feature row paired with its target, so the caller can pass the splits
    straight into a training loop.
    """
    if len(features) != len(targets):
        raise ValueError("features and targets must be the same length")

    random.seed(seed)
    features = list(features)
    targets = list(targets)
    random.shuffle(features)
    random.shuffle(targets)

    n_val = int(len(features) * val_fraction)
    val_x = features[:n_val]
    val_y = targets[:n_val]
    train_x = features[n_val:]
    train_y = targets[n_val:]
    return train_x, train_y, val_x, val_y


def num_batches(data, batch_size=DEFAULT_BATCH_SIZE):
    """Number of minibatches :func:`iter_batches` will yield for ``data``."""
    return len(data) // batch_size


def iter_batches(features, targets, batch_size=DEFAULT_BATCH_SIZE, shuffle=True):
    """Yield ``(batch_x, batch_y)`` pairs walking the dataset once.

    When ``shuffle`` is set the row order is randomised each epoch, which
    keeps consecutive batches from correlating on sorted data.
    """
    order = list(range(len(features)))
    if shuffle:
        random.Random().shuffle(order)

    for start in range(0, len(order) - batch_size, batch_size):
        rows = order[start:start + batch_size]
        yield [features[i] for i in rows], [targets[i] for i in rows]


def standardize(values, transforms=[]):
    """Return ``values`` rescaled to zero mean and unit variance.

    ``transforms`` collects the ``(mean, std)`` pair that was applied, so a
    caller can replay the same scaling on a validation split later.
    """
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)

    transforms.append((mean, std))
    for i in range(len(values)):
        values[i] = (values[i] - mean) / std
    return values


class AverageMeter:
    """Track a running mean, e.g. the loss over one epoch."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        """Fold ``value``, observed over ``n`` samples, into the average."""
        self.total += value
        self.count += n

    @property
    def average(self):
        """Mean of everything folded in so far, or 0.0 when empty."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self):
        self.total = 0.0
        self.count = 0
