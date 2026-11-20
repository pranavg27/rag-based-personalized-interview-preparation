from __future__ import annotations

import re
import uuid
from datetime import date, datetime, timezone
from typing import Any

ID_PATTERN = r"^[0-9a-f]{32}$"
_THREAD_RE = re.compile(r"^user_(?P<user>[0-9a-f]{32})_interview_(?P<interview>[0-9a-f]{32})$")


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_thread_id(user_id: str, interview_id: str) -> str:
    return f"user_{user_id}_interview_{interview_id}"


def validate_thread_id(thread_id: str, user_id: str, interview_id: str | None = None) -> bool:
    """A thread id is valid only if it embeds the *authenticated* user (and interview)."""
    m = _THREAD_RE.match(thread_id or "")
    if not m or m.group("user") != user_id:
        return False
    return interview_id is None or m.group("interview") == interview_id


def serialize(obj: Any) -> Any:
    """Make Mongo documents JSON-safe: `_id`->`id`, datetimes->ISO strings."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out["id" if k == "_id" else k] = serialize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)
