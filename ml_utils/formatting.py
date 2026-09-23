"""Formatting helpers for notebook output."""


def format_duration(seconds: float) -> str:
    """Render an elapsed time the way a training loop should print it.

    Sub-second values are shown in milliseconds; longer ones are broken into
    hours, minutes and seconds, dropping leading units that would be zero.

        >>> format_duration(0.35)
        '350ms'
        >>> format_duration(65)
        '1m 05s'
        >>> format_duration(3725)
        '1h 02m 05s'
    """
    if seconds < 0:
        raise ValueError(f"duration must be non-negative, got {seconds}")

    milliseconds = round(seconds * 1000)
    if milliseconds < 1000:
        return f"{milliseconds}ms"

    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
