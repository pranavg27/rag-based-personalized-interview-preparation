from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.dependencies import CurrentUser, Svc
from app.schemas.content import ChatRequest
from app.utils.sse import SSE_HEADERS, sse_stream

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask")
async def ask(body: ChatRequest, user: CurrentUser, svc: Svc):
    """Non-streaming grounded answer over the user's own documents."""
    return await svc.rag.answer(user["_id"], body.question, body.skill_category)


@router.post("/stream")
async def ask_stream(body: ChatRequest, request: Request, user: CurrentUser, svc: Svc):
    events = svc.rag.answer_stream(user["_id"], body.question, body.skill_category)
    return StreamingResponse(sse_stream(request, events), media_type="text/event-stream", headers=SSE_HEADERS)
