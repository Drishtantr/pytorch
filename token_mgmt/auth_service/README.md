# Auth service

A small API key + refresh token service built on top of the JWT experiments in
`token_mgmt/jwt/`. It covers the pieces a real service needs beyond a bare
signed token: durable key storage, revocation, refresh token rotation, scope
checks and rate limiting.

## Design

| Concern | Approach |
| --- | --- |
| API keys | Opaque `tmk_<32>` strings, generated from a CSPRNG. |
| Key storage | Only a salted SHA-256 digest is persisted; the raw key is shown once at creation. |
| Comparison | Digests are compared in constant time so key checks do not leak length or prefix information. |
| Revocation | `DELETE /api-keys/{owner}` marks every key for that owner revoked. Revocation takes effect immediately, on the very next request. |
| Refresh tokens | Single use. Presenting a refresh token invalidates it and returns its replacement, so a stolen token is usable at most once and reuse is detectable. |
| Access tokens | HS256 JWTs, 15 minute TTL, with a 30 second clock-skew allowance on verification. |
| Scopes | `check_access` requires **all** of the requested scopes to be present. |
| Rate limiting | Sliding window, 100 requests/minute per caller, applied before authentication so that unauthenticated floods are cheap to shed. |

## Usage

```bash
uvicorn token_mgmt.auth_service.app:app --reload

# Get an access token and a refresh token
curl -X POST 'localhost:8000/login?owner=alice&role=service'

# Mint an API key (requires a valid bearer token)
curl -X POST 'localhost:8000/api-keys?owner=alice&scopes=read,write' \
  -H "Authorization: Bearer $ACCESS"

# Call a protected endpoint
curl localhost:8000/secure -H "X-API-Key: $KEY" -H "X-Required-Scope: read"
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `AUTH_SECRET` | `apple` | HMAC signing secret for access tokens. |

## Tests

```bash
pytest tests/test_auth_service.py
```
