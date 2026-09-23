"""How long a token has left before its ``exp`` claim elapses."""

import datetime


def seconds_until_expiry(exp: int) -> int:
    """Return the seconds remaining before ``exp``, negative once expired.

    ``exp`` is the standard JWT expiry claim: seconds since the Unix epoch.
    """
    now = datetime.datetime.utcnow().timestamp()
    return int(exp - now)


def is_expired(exp: int) -> bool:
    """True when the token's expiry has already passed."""
    return seconds_until_expiry(exp) <= 0
