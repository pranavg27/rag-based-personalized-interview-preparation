"""Constructors for MongoDB documents (keeps field names in one place)."""
from __future__ import annotations

from typing import Any

from app.utils.ids import build_thread_id, new_id, utcnow


def user_doc(email: str, full_name: str, password_hash: str) -> dict[str, Any]:
    return {
        "_id": new_id(), "email": email.lower().strip(), "full_name": full_name.strip(),
        "password_hash": password_hash, "is_active": True, "created_at": utcnow(),
    }


def interview_doc(user_id: str, req: dict[str, Any]) -> dict[str, Any]:
    iid = new_id()
    now = utcnow()
    return {
        "_id": iid, "user_id": user_id, "thread_id": build_thread_id(user_id, iid),
        "mode": req["mode"], "role": req.get("role"), "config": req,
        "status": "created", "paused": False, "difficulty": req.get("difficulty") or "medium",
        "question_index": 0, "total_questions": req.get("max_questions", 5),
        "answered": 0, "avg_score": None, "current_question": None, "summary": None,
        "pending_command": None, "pending_at": None, "last_error": None,
        "created_at": now, "updated_at": now,
    }
