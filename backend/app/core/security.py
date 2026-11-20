from __future__ import annotations

from datetime import timedelta

import jwt
from pwdlib import PasswordHash

from app.config import get_settings
from app.errors import UnauthorizedError
from app.utils.ids import utcnow

_hasher = PasswordHash.recommended()  # Argon2id
_DUMMY_HASH = _hasher.hash("dummy-password-for-timing")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, hashed: str | None) -> bool:
    """Verifies against a dummy hash when the user does not exist (uniform timing)."""
    try:
        ok = _hasher.verify(password, hashed or _DUMMY_HASH)
    except Exception:
        return False
    return bool(ok and hashed is not None)


def create_access_token(user_id: str) -> tuple[str, int]:
    s = get_settings()
    expires = timedelta(minutes=s.access_token_expire_minutes)
    now = utcnow()
    token = jwt.encode(
        {"sub": user_id, "iat": now, "exp": now + expires, "type": "access"},
        s.jwt_secret_key,
        algorithm=s.jwt_algorithm,
    )
    return token, int(expires.total_seconds())


def decode_access_token(token: str) -> str:
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret_key, algorithms=[s.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise UnauthorizedError("Invalid or expired token.") from exc
    if payload.get("type") != "access" or not payload.get("sub"):
        raise UnauthorizedError("Invalid token.")
    return str(payload["sub"])
