from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.dependencies import CurrentUser, Svc
from app.errors import BadRequestError
from app.schemas.interview import (AnswerRequest, ControlRequest, InterviewStartRequest, NextRequest,
                                   QueuedResponse)
from app.utils.sse import SSE_HEADERS, sse_stream

router = APIRouter(prefix="/interviews", tags=["interviews"])


async def _dispatch(svc, user_id: str, interview_id: str, stream: bool, stream_path: str) -> QueuedResponse:
    """Command is already queued. Stream -> client opens GET /stream; otherwise run to the next pause point."""
    if stream:
        return QueuedResponse(queued=True, stream_url=f"/interviews/{interview_id}/stream",
                              interview=await svc.interviews.view(user_id, interview_id))
    return QueuedResponse(queued=False, interview=await svc.interviews.run_blocking(user_id, interview_id))


@router.post("/start", response_model=QueuedResponse, status_code=201)
async def start(body: InterviewStartRequest, user: CurrentUser, svc: Svc):
    req = body.model_dump(mode="json", exclude={"stream"})
    doc = await svc.interviews.create(user["_id"], req)
    return await _dispatch(svc, user["_id"], doc["_id"], body.stream, "stream")


@router.get("")
async def list_interviews(user: CurrentUser, svc: Svc):
    return await svc.interviews.list(user["_id"])


@router.get("/{interview_id}")
async def get_interview(interview_id: str, user: CurrentUser, svc: Svc):
    return await svc.interviews.view(user["_id"], interview_id)


@router.get("/{interview_id}/answers")
async def get_answers(interview_id: str, user: CurrentUser, svc: Svc):
    return await svc.interviews.answers(user["_id"], interview_id)


@router.post("/{interview_id}/answer", response_model=QueuedResponse)
async def answer(interview_id: str, body: AnswerRequest, user: CurrentUser, svc: Svc):
    await svc.interviews.queue_command(user["_id"], interview_id, "resume", {
        "action": "answer", "answer": body.answer, "additional_context": body.additional_context})
    return await _dispatch(svc, user["_id"], interview_id, body.stream, "stream")


@router.post("/{interview_id}/next", response_model=QueuedResponse)
async def next_step(interview_id: str, body: NextRequest, user: CurrentUser, svc: Svc):
    """Human-in-the-loop decisions: approve skills/plan, accept/reject/re-evaluate feedback, skip, retry."""
    if body.decision == "retry":
        await svc.interviews.queue_command(user["_id"], interview_id, "continue")
    else:
        value: dict = {"action": body.decision}
        if body.skills is not None:
            value["skills"] = body.skills
        if body.plan is not None:
            value["plan"] = [p.model_dump(mode="json") for p in body.plan]
        if body.note:
            value["note"] = body.note
        await svc.interviews.queue_command(user["_id"], interview_id, "resume", value)
    return await _dispatch(svc, user["_id"], interview_id, body.stream, "stream")


@router.post("/{interview_id}/pause")
async def pause(interview_id: str, user: CurrentUser, svc: Svc):
    return await svc.interviews.pause(user["_id"], interview_id)


@router.post("/{interview_id}/resume")
async def resume(interview_id: str, user: CurrentUser, svc: Svc):
    """Un-pause and return the exact point to continue from (last unanswered question / pending HITL step)."""
    return await svc.interviews.unpause(user["_id"], interview_id)


@router.post("/{interview_id}/end", response_model=QueuedResponse)
async def end(interview_id: str, body: ControlRequest, user: CurrentUser, svc: Svc):
    if not await svc.interviews.can_end_via_graph(user["_id"], interview_id):
        return QueuedResponse(queued=False, interview=await svc.interviews.end_without_graph(user["_id"], interview_id))
    await svc.interviews.queue_command(user["_id"], interview_id, "resume", {"action": "end"})
    return await _dispatch(svc, user["_id"], interview_id, body.stream, "stream")


@router.get("/{interview_id}/stream")
async def stream(interview_id: str, request: Request, user: CurrentUser, svc: Svc):
    """SSE: executes the queued command and streams tokens/status/interrupt events."""
    doc, command = await svc.interviews.claim_command(user["_id"], interview_id)
    events = svc.interviews.run_command(doc, command)
    return StreamingResponse(sse_stream(request, events), media_type="text/event-stream", headers=SSE_HEADERS)
