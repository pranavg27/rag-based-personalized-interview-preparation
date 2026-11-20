"""Orchestrates interviews: ownership checks, thread ids, HITL command validation, graph runs."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from langgraph.types import Command

from app.core.tracing import run_config
from app.database import mongo as m
from app.errors import BadRequestError, ConflictError, ForbiddenError, NotFoundError
from app.graph.nodes import GraphDeps
from app.graph.workflow import build_workflow
from app.models.documents import interview_doc
from app.services.container import Services
from app.services.interview_repo import InterviewRepo
from app.utils.ids import serialize, utcnow, validate_thread_id

log = logging.getLogger(__name__)

# which decisions are valid for each interrupt kind
ALLOWED: dict[str, set[str]] = {
    "review_skills": {"approve", "end"},
    "approve_plan": {"approve", "reject", "end"},
    "await_answer": {"answer", "skip", "end"},
    "review_feedback": {"accept", "reject", "reevaluate", "end"},
}
PENDING_TTL = timedelta(seconds=90)
TERMINAL = {"completed", "ended"}


class InterviewService:
    def __init__(self, services: Services, checkpointer: Any):
        self.s = services
        self.repo = InterviewRepo()
        self.checkpointer = checkpointer
        self.graph = build_workflow(
            GraphDeps(llm=services.llm, memory=services.memory, personalization=services.personalization,
                      evaluation=services.evaluation, repo=self.repo, mcp=services.mcp, settings=services.settings),
            checkpointer)

    # ------------------------------------------------------------------ helpers
    @property
    def col(self):
        return m.get_db()[m.INTERVIEWS]

    def _config(self, doc: dict) -> dict[str, Any]:
        if not validate_thread_id(doc["thread_id"], doc["user_id"], doc["_id"]):
            raise ForbiddenError("Thread does not belong to this user.")
        cfg = run_config("interview_turn", doc["user_id"], doc["_id"], mode=doc["mode"])
        cfg["configurable"] = {"thread_id": doc["thread_id"]}
        cfg["recursion_limit"] = 100
        return cfg

    async def _get_doc(self, user_id: str, interview_id: str) -> dict:
        doc = await self.col.find_one({"_id": interview_id, "user_id": user_id})
        if not doc:
            raise NotFoundError("Interview not found.")
        return doc

    async def _snapshot(self, doc: dict):
        return await self.graph.aget_state(self._config(doc))

    @staticmethod
    def _interrupts(snapshot) -> list[dict]:
        return [i.value for t in (snapshot.tasks or []) for i in (t.interrupts or []) if isinstance(i.value, dict)]

    def _status(self, doc: dict, snapshot) -> str:
        if doc.get("paused"):
            return "paused"
        values = snapshot.values or {}
        if not values:
            return "created"
        if not snapshot.next:
            return "ended" if values.get("end_requested") else "completed"
        if self._interrupts(snapshot):
            return "awaiting_input"
        return "needs_retry"

    async def _sync(self, doc: dict, snapshot) -> dict:
        v = snapshot.values or {}
        status = self._status(doc, snapshot)
        scores = v.get("scores", [])
        upd = {"status": status, "difficulty": v.get("difficulty", doc.get("difficulty")),
               "question_index": v.get("question_index", 0), "answered": len(scores),
               "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
               "current_question": v.get("current_question"), "updated_at": utcnow(), "last_error": None}
        if v.get("plan"):
            upd["total_questions"] = len(v["plan"])
        if v.get("summary"):
            upd["summary"] = v["summary"]
        await self.col.update_one({"_id": doc["_id"], "user_id": doc["user_id"]}, {"$set": upd})
        return {**doc, **upd}

    async def view(self, user_id: str, interview_id: str) -> dict[str, Any]:
        doc = await self._get_doc(user_id, interview_id)
        snap = await self._snapshot(doc)
        v = snap.values or {}
        waiting = (self._interrupts(snap) or [None])[0]
        status = self._status(doc, snap)
        base = {k: doc.get(k) for k in ("_id", "mode", "role", "thread_id", "paused", "created_at", "updated_at",
                                         "last_error", "pending_command")}
        state = {
            "question_index": v.get("question_index", 0), "total_questions": len(v.get("plan", [])) or doc.get("total_questions"),
            "difficulty": v.get("difficulty"), "plan": v.get("plan", []), "skills": v.get("skills", []),
            "jd_skills": v.get("jd_skills", []), "current_question": v.get("current_question"),
            "scores": v.get("scores", []), "difficulty_history": v.get("difficulty_history", []),
            "messages": v.get("messages", [])[-40:], "summary": v.get("summary"),
            "weak_topics": v.get("weak_topics", []),
        }
        return serialize({**base, "status": status, "waiting_for": waiting, "state": state,
                          "has_pending_command": bool(doc.get("pending_command"))})

    async def list(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = await self.col.find({"user_id": user_id}).sort("created_at", -1).to_list(limit)
        keep = ("_id", "mode", "role", "status", "paused", "avg_score", "answered", "total_questions",
                "difficulty", "created_at", "updated_at")
        return [serialize({k: r.get(k) for k in keep}) for r in rows]

    async def answers(self, user_id: str, interview_id: str) -> list[dict]:
        await self._get_doc(user_id, interview_id)
        return serialize(await self.repo.list_answers(user_id, interview_id))

    # ------------------------------------------------------------------ create
    async def create(self, user_id: str, req: dict[str, Any]) -> dict:
        jd_text = req.get("job_description") or ""
        if req.get("jd_document_id"):
            d = await m.get_db()[m.DOCUMENTS].find_one({"_id": req["jd_document_id"], "user_id": user_id})
            if not d or d.get("doc_type") != "job_description":
                raise NotFoundError("Job description document not found.")
            jd_text = d.get("text", "")
        if req.get("resume_id"):
            await self.s.resumes.get(user_id, req["resume_id"])  # ownership check
        prefs = await self.s.memory.get_preferences(user_id)
        role = req.get("role") or prefs.get("preferred_role")
        req = {**req, "role": role}
        doc = interview_doc(user_id, {k: v for k, v in req.items() if k != "job_description"})
        initial = {
            "interview_id": doc["_id"], "user_id": user_id, "thread_id": doc["thread_id"], "mode": req["mode"],
            "role": role or "", "resume_id": req.get("resume_id") or "", "requested_skills": req.get("skills", []),
            "jd_text": jd_text[:20000], "requested_difficulty": req.get("difficulty") or "",
            "max_questions": req.get("max_questions", 5), "hitl": req.get("hitl") or {},
            "question_index": 0, "previous_questions": [], "answers": [], "evaluations": [], "scores": [],
            "difficulty_history": [], "messages": [], "end_requested": False, "plan_revision": 0,
        }
        doc["pending_command"] = {"kind": "start", "input": initial}
        doc["pending_at"] = utcnow()
        await self.col.insert_one(doc)
        if role:
            await self.s.memory.set_preferences(user_id, {"preferred_role": role})
        return doc

    # ------------------------------------------------------------------ commands
    async def queue_command(self, user_id: str, interview_id: str, kind: str, value: dict | None = None) -> dict:
        doc = await self._get_doc(user_id, interview_id)
        if doc.get("paused"):
            raise ConflictError("Interview is paused. Resume it first.")
        snap = await self._snapshot(doc)
        status = self._status(doc, snap)
        if status in TERMINAL:
            raise ConflictError(f"Interview already {status}.")
        if kind == "resume":
            waiting = (self._interrupts(snap) or [None])[0]
            if not waiting:
                raise ConflictError("The interview is not waiting for input right now.")
            action = (value or {}).get("action")
            if action not in ALLOWED.get(waiting["kind"], set()):
                raise ConflictError(f"'{action}' is not valid while the interview is waiting for '{waiting['kind']}'.")
        elif kind == "continue":
            if status != "needs_retry":
                raise ConflictError("Nothing to retry.")
        elif kind != "start":
            raise BadRequestError("Unknown command.")
        now = utcnow()
        res = await self.col.update_one(
            {"_id": interview_id, "user_id": user_id,
             "$or": [{"pending_command": None}, {"pending_at": {"$lt": now - PENDING_TTL}}]},
            {"$set": {"pending_command": {"kind": kind, "value": value or {}}, "pending_at": now}})
        if res.matched_count == 0:
            raise ConflictError("Another command is already pending for this interview.")
        return doc

    async def queue_start_if_needed(self, user_id: str, interview_id: str) -> None:
        return None  # `create` stores the start command already

    async def claim_command(self, user_id: str, interview_id: str) -> tuple[dict, dict]:
        before = await self.col.find_one_and_update(
            {"_id": interview_id, "user_id": user_id, "pending_command": {"$ne": None}},
            {"$set": {"pending_command": None}})
        if not before:
            await self._get_doc(user_id, interview_id)
            raise ConflictError("No pending command. POST an action first, then open the stream.")
        return before, before["pending_command"]

    async def run_command(self, doc: dict, command: dict) -> AsyncIterator[dict[str, Any]]:
        cfg = self._config(doc)
        kind = command["kind"]
        inp: Any
        if kind == "start":
            inp = command["input"]
        elif kind == "resume":
            inp = Command(resume=command["value"])
        else:
            inp = None
        yield {"type": "run_start", "interview_id": doc["_id"], "command": kind}
        try:
            async for mode, chunk in self.graph.astream(inp, cfg, stream_mode=["custom", "updates"]):
                if mode == "custom":
                    yield chunk
                else:
                    for node in chunk:
                        if node != "__interrupt__":
                            yield {"type": "node_done", "node": node}
        except asyncio.CancelledError:
            log.info("run cancelled for %s (client disconnected)", doc["_id"])
            await asyncio.shield(self.col.update_one({"_id": doc["_id"]}, {"$set": {"updated_at": utcnow()}}))
            raise
        except Exception as exc:
            await self.col.update_one({"_id": doc["_id"], "user_id": doc["user_id"]},
                                      {"$set": {"last_error": str(exc)[:300], "updated_at": utcnow()}})
            raise
        snap = await self._snapshot(doc)
        await self._sync(doc, snap)
        view = await self.view(doc["user_id"], doc["_id"])
        yield {"type": "state", "view": view}
        if view["waiting_for"]:
            yield {"type": "interrupt", **view["waiting_for"]}
        else:
            yield {"type": "complete", "status": view["status"]}

    async def run_blocking(self, user_id: str, interview_id: str) -> dict:
        doc, command = await self.claim_command(user_id, interview_id)
        async for _ in self.run_command(doc, command):
            pass
        return await self.view(user_id, interview_id)

    # ------------------------------------------------------------------ pause / resume / end
    async def pause(self, user_id: str, interview_id: str) -> dict:
        doc = await self._get_doc(user_id, interview_id)
        status = self._status(doc, await self._snapshot(doc))
        if status in TERMINAL:
            raise ConflictError(f"Interview already {status}.")
        await self.col.update_one({"_id": interview_id, "user_id": user_id},
                                  {"$set": {"paused": True, "status": "paused", "updated_at": utcnow()}})
        return await self.view(user_id, interview_id)

    async def unpause(self, user_id: str, interview_id: str) -> dict:
        doc = await self._get_doc(user_id, interview_id)
        if not doc.get("paused"):
            raise ConflictError("Interview is not paused.")
        await self.col.update_one({"_id": interview_id, "user_id": user_id},
                                  {"$set": {"paused": False, "updated_at": utcnow()}})
        doc["paused"] = False
        await self._sync(doc, await self._snapshot(doc))
        return await self.view(user_id, interview_id)

    async def end_without_graph(self, user_id: str, interview_id: str) -> dict:
        await self.col.update_one({"_id": interview_id, "user_id": user_id},
                                  {"$set": {"status": "ended", "paused": False, "updated_at": utcnow()}})
        return await self.view(user_id, interview_id)

    async def can_end_via_graph(self, user_id: str, interview_id: str) -> bool:
        doc = await self._get_doc(user_id, interview_id)
        if doc.get("paused"):
            doc = {**doc, "paused": False}
            await self.col.update_one({"_id": interview_id}, {"$set": {"paused": False}})
        return bool(self._interrupts(await self._snapshot(doc)))
