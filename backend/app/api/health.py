from __future__ import annotations

import os

from fastapi import APIRouter

from app.config import get_settings
from app.database import mongo as m

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    s = get_settings()
    try:
        await m.get_db().command("ping")
        mongo = "ok"
    except Exception as exc:
        mongo = f"error: {type(exc).__name__}"
    return {"status": "ok" if mongo == "ok" else "degraded", "mongodb": mongo, "model": s.llm_model_id,
            "llm_configured": bool(s.hf_token), "vector_store": s.vector_store,
            "reranker_enabled": s.reranker_enabled, "mcp_mode": s.mcp_mode, "checkpointer": s.checkpointer,
            "langsmith_tracing": os.environ.get("LANGSMITH_TRACING") == "true"}
