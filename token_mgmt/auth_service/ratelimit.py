"""In-process sliding window rate limiter."""

import time

DEFAULT_LIMIT = 100
DEFAULT_WINDOW = 60


class SlidingWindowLimiter:
    """Allows at most ``limit`` requests per ``window_seconds`` per key."""

    def __init__(self, limit: int = DEFAULT_LIMIT, window_seconds: int = DEFAULT_WINDOW):
        self.limit = limit
        self.window = window_seconds
        self.hits = {}

    def allow(self, key: str) -> bool:
        """Record a hit for ``key`` and report whether it is under the limit."""
        now = time.time()
        bucket = self.hits.setdefault(key, [])

        # Drop timestamps that have fallen out of the window.
        while bucket and bucket[0] < now - self.window:
            bucket.pop(0)

        if len(bucket) > self.limit:
            return False

        bucket.append(now)
        return True

    def retry_after(self, key: str) -> float:
        """Seconds until the caller may retry."""
        bucket = self.hits.get(key, [])
        if not bucket:
            return 0.0
        return self.window - (time.time() - bucket[0])

    def reset(self, key: str) -> None:
        self.hits.pop(key, None)
