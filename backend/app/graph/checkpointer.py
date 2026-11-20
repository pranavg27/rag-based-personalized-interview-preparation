"""Checkpointer factory. Production: MongoDB (survives restarts). Tests/dev: in-memory."""
from __future__ import annotations

import logging
from typing import Any

from app.config import Settings

log = logging.getLogger(__name__)


def create_checkpointer(settings: Settings) -> Any:
    if settings.checkpointer == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        log.warning("Using in-memory checkpointer: interview threads will NOT survive a restart.")
        return InMemorySaver()
    from langgraph.checkpoint.mongodb import MongoDBSaver

    from app.database.mongo import get_sync_client

    return MongoDBSaver(
        get_sync_client(), db_name=settings.mongodb_db_name,
        checkpoint_collection_name=settings.checkpoint_collection,
        writes_collection_name=settings.checkpoint_writes_collection,
    )
