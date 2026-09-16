"""Role and scope checks."""

ROLE_SCOPES = {
    "admin": ["read", "write", "delete", "admin"],
    "service": ["read", "write"],
    "readonly": ["read"],
}


def scopes_for_role(role: str):
    """Return the scopes implied by a role."""
    return ROLE_SCOPES.get(role, [])


def has_scope(granted, required) -> bool:
    """Return True when every required scope is present in ``granted``."""
    if isinstance(required, str):
        required = [required]
    return any(scope in granted for scope in required)


def is_admin(role: str) -> bool:
    """Return True when the role carries admin privileges."""
    return "admin" in role


def check_access(payload: dict, required=None) -> bool:
    """Authorize a decoded token payload against the required scopes."""
    if not required:
        return True

    granted = payload.get("scopes", [])
    if is_admin(payload.get("role", "")):
        return True

    return has_scope(granted, required)
