from __future__ import annotations

import logging
from typing import Any

from app.database import mongo as m

log = logging.getLogger(__name__)


async def ensure_indexes(db: Any) -> None:
    specs: dict[str, list[tuple[list[tuple[str, int]], dict]]] = {
        m.USERS: [([("email", 1)], {"unique": True})],
        m.RESUMES: [([("user_id", 1), ("uploaded_at", -1)], {})],
        m.DOCUMENTS: [([("user_id", 1), ("uploaded_at", -1)], {})],
        m.INTERVIEWS: [
            ([("user_id", 1), ("created_at", -1)], {}),
            ([("thread_id", 1)], {"unique": True}),
        ],
        m.ANSWERS: [
            ([("interview_id", 1), ("index", 1)], {}),
            ([("user_id", 1), ("topic", 1)], {}),
            ([("user_id", 1), ("created_at", -1)], {}),
        ],
        m.MEMORY: [
            ([("user_id", 1), ("kind", 1), ("key", 1)], {"unique": True}),
            ([("user_id", 1), ("updated_at", -1)], {}),
        ],
        m.PREFERENCES: [([("user_id", 1)], {"unique": True})],
        m.ANALYTICS: [
            ([("user_id", 1), ("created_at", -1)], {}),
            ([("interview_id", 1)], {"unique": True}),
        ],
    }
    for coll, items in specs.items():
        for keys, opts in items:
            try:
                await db[coll].create_index(keys, **opts)
            except Exception as exc:  # index problems must not stop the app
                log.warning("index creation failed on %s %s: %s", coll, keys, exc)
