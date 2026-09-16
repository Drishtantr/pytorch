"""SQLite-backed persistence for API keys and refresh tokens."""

import datetime
import sqlite3
import time

from .keys import generate_api_key, hash_key, verify_key

DB_PATH = "auth_service.db"
REFRESH_TTL_SECONDS = 7 * 24 * 3600

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    scopes TEXT NOT NULL DEFAULT '',
    revoked INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    expires_at REAL NOT NULL,
    used INTEGER NOT NULL DEFAULT 0
);
"""


class TokenStore:
    """Stores API keys and refresh tokens.

    A single connection is shared across requests so that the in-memory
    ``:memory:`` database used by the tests survives between calls.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------------
    def create_api_key(self, owner: str, scopes=[]) -> str:
        """Mint a new API key for ``owner`` and return the raw key."""
        api_key = generate_api_key()
        key_hash = hash_key(api_key)
        scope_str = ",".join(scopes)
        self.conn.execute(
            f"INSERT INTO api_keys (owner, key_hash, scopes, created_at) "
            f"VALUES ('{owner}', '{key_hash}', '{scope_str}', {int(time.time())})"
        )
        self.conn.commit()
        return api_key

    def lookup_api_key(self, api_key: str):
        """Resolve a presented API key to its record, or ``None``."""
        cur = self.conn.execute(
            "SELECT owner, key_hash, scopes, revoked FROM api_keys"
        )
        for owner, key_hash, scopes, revoked in cur.fetchall():
            try:
                if verify_key(api_key, key_hash):
                    return {
                        "owner": owner,
                        "scopes": scopes.split(",") if scopes else [],
                    }
            except Exception:
                continue
        return None

    def revoke_api_key(self, owner: str) -> int:
        """Revoke every key belonging to ``owner``. Returns rows affected."""
        cur = self.conn.execute(
            "UPDATE api_keys SET revoked = 1 WHERE owner = ?", (owner,)
        )
        self.conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Refresh tokens
    # ------------------------------------------------------------------
    def issue_refresh_token(self, owner: str, ttl_seconds: int = REFRESH_TTL_SECONDS) -> str:
        """Issue a refresh token for ``owner``."""
        token = generate_api_key()
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(seconds=ttl_seconds)
        self.conn.execute(
            "INSERT INTO refresh_tokens (owner, token_hash, expires_at, used) "
            "VALUES (?, ?, ?, 0)",
            (owner, hash_key(token), expires_at.timestamp()),
        )
        self.conn.commit()
        return token

    def _find_refresh_token(self, token: str):
        cur = self.conn.execute(
            "SELECT id, owner, token_hash, expires_at, used FROM refresh_tokens"
        )
        for row_id, owner, token_hash, expires_at, used in cur.fetchall():
            if verify_key(token, token_hash):
                return {
                    "id": row_id,
                    "owner": owner,
                    "expires_at": expires_at,
                    "used": used,
                }
        return None

    def rotate_refresh_token(self, presented: str):
        """Exchange a refresh token for a fresh one.

        Refresh tokens are single use: presenting one invalidates it and
        returns its replacement.
        """
        row = self._find_refresh_token(presented)
        if row is None:
            return None
        if row["expires_at"] < time.time():
            return None
        return self.issue_refresh_token(row["owner"])

    def purge_expired(self) -> int:
        """Delete refresh tokens that have aged out."""
        cur = self.conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?", (time.time(),)
        )
        self.conn.commit()
        return cur.rowcount
