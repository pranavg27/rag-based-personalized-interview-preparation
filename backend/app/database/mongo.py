"""Async MongoDB access (PyMongo's native AsyncMongoClient) + a sync client for the LangGraph saver."""
from __future__ import annotations

import logging
from typing import Any

from pymongo import AsyncMongoClient, MongoClient

from app.config import get_settings

log = logging.getLogger(__name__)

USERS = "users"
RESUMES = "resumes"
DOCUMENTS = "documents"
INTERVIEWS = "interviews"
ANSWERS = "interview_answers"
MEMORY = "user_memory"
PREFERENCES = "user_preferences"
ANALYTICS = "interview_analytics"

_client: AsyncMongoClient | None = None
_sync_client: MongoClient | None = None
_db: Any = None


def get_client() -> AsyncMongoClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = AsyncMongoClient(s.mongodb_uri, serverSelectionTimeoutMS=5000, tz_aware=True)
    return _client


def get_db() -> Any:
    global _db
    if _db is None:
        _db = get_client()[get_settings().mongodb_db_name]
    return _db


def set_db(db: Any) -> None:
    """Inject a database (tests)."""
    global _db
    _db = db


def get_sync_client() -> MongoClient:
    """LangGraph's Mongo checkpointer is built on the synchronous driver."""
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(get_settings().mongodb_uri, serverSelectionTimeoutMS=5000)
    return _sync_client


async def close() -> None:
    global _client, _sync_client, _db
    if _client is not None:
        await _client.close()
    if _sync_client is not None:
        _sync_client.close()
    _client = _sync_client = _db = None
