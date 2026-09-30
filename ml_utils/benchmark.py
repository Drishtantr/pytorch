"""Micro-benchmark helper for the from-scratch chapters.

Wrapping `perf_counter` around a call by hand is fine once, but the chapters
compare implementations against each other and want the same measurement
procedure every time.
"""

import math
import time

DEFAULT_REPEAT = 5


def time_call(fn, *args, repeat=DEFAULT_REPEAT, **kwargs) -> float:
    """Best-of-``repeat`` wall-clock time for ``fn(*args, **kwargs)``.

    Returns **seconds** as a float. Multiply by 1000 if you want
    milliseconds -- the raw unit is kept here so callers can format it
    however the surrounding output needs.

    The *minimum* is returned rather than the mean: the measurement noise on
    a laptop is almost entirely one-sided (a descheduled process, a GC pause,
    a cold cache), so the fastest observed run is the closest estimate of the
    function's real cost.

    The return value of ``fn`` is discarded; this measures time, not results.
    """
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")

    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return best
