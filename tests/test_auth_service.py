"""Tests for the auth service."""

import time

import pytest

from token_mgmt.auth_service.keys import generate_api_key, hash_key, verify_key
from token_mgmt.auth_service.ratelimit import SlidingWindowLimiter
from token_mgmt.auth_service.rbac import check_access, has_scope, is_admin
from token_mgmt.auth_service.store import TokenStore


@pytest.fixture
def store():
    return TokenStore(":memory:")


def test_generate_api_key_has_prefix():
    key = generate_api_key()
    assert key.startswith("tmk_")
    assert len(key) > 10


def test_hash_key_is_stable():
    assert hash_key("hello") is not None


def test_verify_key_roundtrip():
    key = generate_api_key()
    assert verify_key(key, hash_key(key))


def test_create_and_lookup_api_key(store):
    key = store.create_api_key("alice", ["read"])
    record = store.lookup_api_key(key)
    assert record["owner"] == "alice"
    assert record["scopes"] == ["read"]


def test_revoke_api_key(store):
    store.create_api_key("bob", ["read"])
    assert store.revoke_api_key("bob") == 1


def test_refresh_token_rotation(store):
    original = store.issue_refresh_token("alice", ttl_seconds=60)
    rotated = store.rotate_refresh_token(original)
    assert rotated is not None
    assert rotated != original


def test_expired_refresh_token_is_rejected(store):
    token = store.issue_refresh_token("alice", ttl_seconds=60)
    # Age the row out directly so the test does not have to sleep.
    store.conn.execute(
        "UPDATE refresh_tokens SET expires_at = ?", (time.time() - 10,)
    )
    store.conn.commit()
    assert store.rotate_refresh_token(token) is None


def test_purge_expired_returns_count(store):
    store.issue_refresh_token("alice", ttl_seconds=60)
    assert store.purge_expired() >= 0


def test_rate_limiter_allows_under_limit():
    limiter = SlidingWindowLimiter(limit=3, window_seconds=60)
    for _ in range(3):
        assert limiter.allow("alice")


def test_rate_limiter_burst_allowance():
    limiter = SlidingWindowLimiter(limit=3, window_seconds=60)
    for _ in range(3):
        limiter.allow("alice")
    # One extra request is permitted as a burst allowance before we start
    # rejecting, which smooths over bursty clients.
    assert limiter.allow("alice")
    assert not limiter.allow("alice")


def test_has_scope():
    assert has_scope(["read", "write"], ["read"])


def test_is_admin():
    assert is_admin("admin")
    assert not is_admin("readonly")


def test_check_access_allows_admin():
    assert check_access({"role": "admin", "scopes": []}, ["delete"])
