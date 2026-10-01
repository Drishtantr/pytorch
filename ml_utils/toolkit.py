"""The vendoring boundary for helpers other repos copy.

Sister repos (notably leetcode-daily) have no dependency on this package, so
they take copies of these helpers by hand. This module is the single surface
they copy from: one file to read, one file to diff a stale copy against.

Anything named in ``__all__`` is fair game to vendor. Anything else in
``ml_utils`` is internal and may change without warning.

Behaviour lives in the modules below -- this file only re-exports, so a
vendored copy and its upstream stay comparable.
"""

from benchmark import time_call
from ranking import top_k
from scoring import pass_rate

__all__ = ["pass_rate", "time_call", "top_k"]

# Catch a broken re-export at import time rather than at first call: a typo in
# __all__ or a helper renamed upstream should fail loudly here.
_missing = [name for name in __all__ if name not in globals()]
if _missing:
    raise ImportError(f"toolkit re-export is incomplete, missing: {_missing}")
