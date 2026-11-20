from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.dependencies import CurrentUser, Svc
from app.utils.ids import serialize

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard")
async def dashboard(user: CurrentUser, svc: Svc):
    return serialize(await svc.analytics.dashboard(user["_id"]))


@router.get("/history")
async def history(user: CurrentUser, svc: Svc, limit: Annotated[int, Query(ge=1, le=100)] = 20,
                  skip: Annotated[int, Query(ge=0)] = 0):
    return await svc.analytics.history(user["_id"], limit, skip)


@router.get("/weak-topics")
async def weak_topics(user: CurrentUser, svc: Svc, limit: Annotated[int, Query(ge=1, le=20)] = 5):
    return await svc.analytics.weak_topics(user["_id"], limit)


@router.get("/performance")
async def performance(user: CurrentUser, svc: Svc):
    return await svc.analytics.performance(user["_id"])


@router.get("/recommendations")
async def recommendations(user: CurrentUser, svc: Svc):
    return await svc.analytics.recommendations(user["_id"])


@router.get("/report")
async def report(user: CurrentUser, svc: Svc):
    """Progress report produced through the analytics MCP server."""
    return await svc.mcp.call("generate_progress_report", {"user_id": user["_id"]})


@router.get("/practice-questions")
async def practice_questions(user: CurrentUser, svc: Svc, count: Annotated[int, Query(ge=1, le=5)] = 2):
    """Practice questions on weak topics: MCP get_weak_topics -> MCP generate_interview_questions."""
    uid = user["_id"]
    weak = (await svc.mcp.call("get_weak_topics", {"user_id": uid, "limit": 3}, default={})).get("weak_topics", [])
    topics = [w["topic"] for w in weak] or ["Python"]
    out = []
    for t in topics:
        res = await svc.mcp.call("generate_interview_questions",
                                 {"user_id": uid, "topic": t, "difficulty": "easy" if weak else "medium", "count": count},
                                 default={"questions": []})
        out.append({"topic": t, "questions": res.get("questions", [])})
    return {"based_on_weak_topics": bool(weak), "practice": out}
