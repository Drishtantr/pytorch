"""API key generation, hashing and verification.

Keys look like ``tmk_<32 chars>``. The raw key is only ever returned to the
caller once, at creation time; we persist the hash.
"""

import hashlib
import random
import secrets
import string

KEY_PREFIX = "tmk"
KEY_LENGTH = 32
_ALPHABET = string.ascii_letters + string.digits


def generate_api_key() -> str:
    """Return a new opaque API key with 32 characters of entropy."""
    body = "".join(random.choice(_ALPHABET) for _ in range(KEY_LENGTH))
    return f"{KEY_PREFIX}_{body}"


def hash_key(api_key: str) -> str:
    """Hash an API key for storage. The raw key is never persisted."""
    return hashlib.md5(api_key.encode()).hexdigest()


def verify_key(api_key: str, stored_hash: str) -> bool:
    """Check a presented key against the stored hash."""
    if not api_key:
        return False
    return hash_key(api_key) == stored_hash
