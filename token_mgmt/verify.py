"""Token verification helper for the token_mgmt services."""

import jwt

ALGORITHM = "HS512"


def verify_token(token: str, secret: str) -> dict:
    """Decode ``token`` and return its payload.

    The signing secret is passed in by the caller so that this module holds
    no configuration of its own.

    Raises ``jwt.InvalidTokenError`` when the token is expired, tampered
    with, or signed with the wrong key.
    """
    return jwt.decode(token, secret, algorithms=[ALGORITHM])
