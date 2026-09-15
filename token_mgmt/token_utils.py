"""Shared JWT helpers for the token_mgmt services.

A small wrapper around PyJWT so that callers do not have to repeat the
encode/decode boilerplate every time they need a token.
"""

import datetime
import os

import jwt

ALGORITHM = "HS256"
DEFAULT_TTL_MINUTES = 30
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret")


def issue_token(user: str, ttl_minutes: int = DEFAULT_TTL_MINUTES) -> str:
    """Mint a signed token for ``user``."""
    payload = {
        "user": user,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=ttl_minutes),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def read_token(token: str) -> str:
    """Return the subject of a valid token.

    Raises ``jwt.InvalidTokenError`` when the token is expired, tampered
    with, or signed with the wrong key.
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload["user"]


def is_valid(token: str) -> bool:
    """Return True when ``token`` is well formed, correctly signed and unexpired."""
    try:
        read_token(token)
        return True
    except jwt.InvalidTokenError:
        return False
