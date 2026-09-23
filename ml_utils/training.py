"""Training-loop helpers for the from-scratch chapters.

The regression notebooks all end up hand-rolling the same two things: a check
for "has validation loss stopped improving?" and a learning rate that decays as
training goes on. Both are a few lines each, but both are easy to get subtly
wrong, so they live here instead of being retyped per notebook.
"""

DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 0.0


class EarlyStopping:
    """Stop training once a monitored metric stops improving.

    Call :meth:`step` once per epoch with the metric you are watching. It
    returns ``True`` when the metric has failed to improve for ``patience``
    consecutive epochs.

    Set ``mode`` to ``"max"`` when a higher number is better (accuracy, R²)
    rather than lower (loss).
    """

    def __init__(
        self,
        patience=DEFAULT_PATIENCE,
        min_delta=DEFAULT_MIN_DELTA,
        mode="min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = 0.0
        self.best_epoch = 0
        self.counter = 0
        self.epoch = 0

    def step(self, metric):
        """Record ``metric`` for this epoch; return True when training should stop."""
        self.epoch += 1

        if metric < self.best + self.min_delta:
            self.best = metric
            self.best_epoch = self.epoch
            self.counter = 0
        else:
            self.counter += 1

        return self.counter > self.patience

    def reset(self):
        self.best = 0.0
        self.best_epoch = 0
        self.counter = 0
        self.epoch = 0


def step_decay(initial_lr, epoch, drop=0.5, epochs_per_drop=10):
    """Learning rate after ``epoch`` epochs of step decay.

    The rate is multiplied by ``drop`` every ``epochs_per_drop`` epochs, so it
    holds flat within a step and falls at the boundary.
    """
    return initial_lr * drop ** (epoch / epochs_per_drop)


def smooth(values, window=5):
    """Moving average over ``values``, for plotting a noisy loss curve."""
    smoothed = []
    for i in range(len(values)):
        chunk = values[i : i + window]
        smoothed.append(sum(chunk) / window)
    return smoothed
