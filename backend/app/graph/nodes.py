"""LangGraph nodes. Each node is a method on InterviewNodes so it can be instantiated with fakes
and tested in isolation. Nodes that wait for a human (interrupt) contain NO side effects before the
interrupt, because LangGraph re-executes the node from its start when the thread resumes."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from langsmith import traceable

from app.config import Settings
from app.graph.state import InterviewState
from app.mcp.client import MCPToolbox
from app.models.enums import DIFFICULTY_LADDER, PLAN_TYPES
from app.services import prompts
from app.services.evaluation_service import EvaluationService
from app.services.interview_repo import InterviewRepo
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.personalization_service import PersonalizationService
from app.services.resume_service import dedupe
from app.utils.ids import new_id

log = logging.getLogger(__name__)


@dataclass
class GraphDeps:
    llm: LLMService
    memory: MemoryService
    personalization: PersonalizationService
    evaluation: EvaluationService
    repo: InterviewRepo
    mcp: MCPToolbox
    settings: Settings


def emit(event: dict[str, Any]) -> None:
    """Send a custom stream event; no-op outside a streaming graph run."""
    try:
        get_stream_writer()(event)
    except Exception:
        pass


def clean_plan(raw: list[dict] | None) -> list[dict[str, Any]]:
    out = []
    for i, p in enumerate((raw or [])[:20]):
        topic = str(p.get("topic", "")).strip()[:80]
        if not topic:
            continue
        typ = p.get("type") if p.get("type") in PLAN_TYPES else "technical"
        diff = p.get("difficulty") if p.get("difficulty") in DIFFICULTY_LADDER else None
        out.append({"index": i, "type": typ, "topic": topic, "difficulty": diff})
    return out


def compute_stats(state: InterviewState) -> dict[str, Any]:
    by_topic: dict[str, list[float]] = {}
    for q, ev in zip(state.get("previous_questions", []), state.get("evaluations", [])):
        if ev and not ev.get("rejected") and not q.get("skipped"):
            by_topic.setdefault(q["topic"], []).append(ev["score"])
    scores = state.get("scores", [])
    topic_scores = {t: round(sum(v) / len(v), 2) for t, v in by_topic.items()}
    ranked = sorted(topic_scores.items(), key=lambda kv: kv[1])
    return {
        "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
        "answered": len(scores),
        "skipped": sum(1 for q in state.get("previous_questions", []) if q.get("skipped")),
        "total_planned": len(state.get("plan", [])),
        "topic_scores": topic_scores,
        "weakest_topics": [t for t, s in ranked[:2] if s < 7],
        "strongest_topics": [t for t, s in ranked[::-1][:2] if s >= 7],
        "difficulty_history": state.get("difficulty_history", []),
        "ended_early": bool(state.get("end_requested")),
    }


class InterviewNodes:
    def __init__(self, deps: GraphDeps):
        self.d = deps

    # ---------------------------------------------------------------- analysis
    async def analyze_resume(self, state: InterviewState) -> dict[str, Any]:
        emit({"type": "node_start", "node": "analyze_resume", "message": "Analysing resume…"})
        rid, uid = state.get("resume_id"), state["user_id"]
        if not rid:
            return {"resume_profile": {}}
        summary = await self.d.mcp.call("summarize_resume", {"user_id": uid, "resume_id": rid}, default={})
        projects = await self.d.mcp.call("analyze_resume_projects", {"user_id": uid, "resume_id": rid}, default={})
        return {"resume_profile": {
            "summary": summary.get("summary", ""), "experience": summary.get("experience", []),
            "education": summary.get("education", []), "projects": projects.get("projects", [])}}

    async def extract_skills(self, state: InterviewState) -> dict[str, Any]:
        emit({"type": "node_start", "node": "extract_skills", "message": "Identifying skills…"})
        uid, rid = state["user_id"], state.get("resume_id")
        skills = list(state.get("requested_skills") or [])
        if rid:
            res = await self.d.mcp.call("extract_resume_skills", {"user_id": uid, "resume_id": rid}, default={})
            skills = dedupe(res.get("skills", []) + skills)
        out: dict[str, Any] = {"skills": skills}
        jd = state.get("jd_text")
        if jd:
            analysis = await self.d.mcp.call("analyze_job_description",
                                             {"user_id": uid, "job_description": jd, "role": state.get("role")}, default={})
            req = await self.d.mcp.call("extract_required_skills", {"job_description": jd}, default={})
            jd_skills = dedupe(analysis.get("required_skills", []) + req.get("skills", []))
            out["jd_skills"] = jd_skills
            out["jd_analysis"] = dict(analysis)
            if rid:
                cmp_ = await self.d.mcp.call("compare_resume_with_job_description",
                                             {"user_id": uid, "resume_id": rid, "job_description": jd}, default={})
                out["jd_analysis"]["comparison"] = cmp_
                if cmp_.get("missing"):  # prioritise gaps
                    out["jd_skills"] = dedupe(cmp_["missing"] + jd_skills)
        return out

    async def review_skills(self, state: InterviewState) -> dict[str, Any]:  # HITL
        decision = interrupt({"kind": "review_skills", "skills": state.get("skills", []),
                              "jd_skills": state.get("jd_skills", []),
                              "jd_analysis": state.get("jd_analysis", {})})
        if decision.get("action") == "end":
            return {"end_requested": True}
        edited = [str(s).strip()[:60] for s in (decision.get("skills") or []) if str(s).strip()]
        if edited:
            await self.d.memory.remember_skills(state["user_id"], edited)
            return {"skills": dedupe(edited)}
        return {}

    async def plan_interview(self, state: InterviewState) -> dict[str, Any]:
        emit({"type": "node_start", "node": "plan_interview", "message": "Planning your interview…"})
        uid = state["user_id"]
        weak = await self.d.mcp.call("get_weak_topics", {"user_id": uid, "limit": 5}, default={})
        weak_names = [w["topic"] for w in weak.get("weak_topics", [])]
        perf = await self.d.mcp.call("get_user_performance", {"user_id": uid}, default={})
        prefs = await self.d.memory.get_preferences(uid)
        difficulty = self.d.personalization.initial_difficulty(
            state.get("requested_difficulty"), prefs.get("preferred_difficulty"), perf.get("overall_avg"))
        revision = state.get("plan_revision", 0)
        plan = self.d.personalization.build_plan(
            mode=state["mode"], role=state.get("role"), requested_skills=state.get("requested_skills", []),
            resume_skills=state.get("skills", []), jd_skills=state.get("jd_skills", []), weak_topics=weak_names,
            num_questions=state.get("max_questions", 5),
            has_projects=bool(state.get("resume_profile", {}).get("projects")), variation=revision)
        return {"plan": plan, "weak_topics": weak_names, "difficulty": difficulty,
                "difficulty_history": [difficulty], "plan_rejected": False}

    async def approve_plan(self, state: InterviewState) -> dict[str, Any]:  # HITL
        decision = interrupt({"kind": "approve_plan", "plan": state["plan"], "difficulty": state.get("difficulty"),
                              "weak_topics": state.get("weak_topics", [])})
        action = decision.get("action")
        if action == "end":
            return {"end_requested": True}
        if action == "reject":
            return {"plan_rejected": True, "plan_feedback": str(decision.get("note") or "")[:300],
                    "plan_revision": state.get("plan_revision", 0) + 1}
        edited = clean_plan(decision.get("plan"))
        return {"plan_rejected": False, "plan": edited or state["plan"]}

    # ---------------------------------------------------------------- question loop
    async def retrieve_knowledge(self, state: InterviewState) -> dict[str, Any]:
        idx = state.get("question_index", 0)
        item = state["plan"][idx]
        topic, uid = item["topic"], state["user_id"]
        emit({"type": "node_start", "node": "retrieve_knowledge", "message": f"Retrieving knowledge on {topic}…"})
        out: dict[str, Any] = {"current_topic": topic, "retrieved_context": "", "sources": [], "asked_before": []}
        if item["type"] in ("behavioral", "followup"):
            return out
        strategy = {"resume_based": "resume", "jd_based": "jd"}.get(state["mode"], "technical")
        query = self._query(strategy, topic, state)
        res = await self.d.mcp.call("search_technical_knowledge",
                                    {"user_id": uid, "query": query, "strategy": strategy}, default={})
        bank = await self.d.mcp.call("retrieve_interview_questions",
                                     {"user_id": uid, "topic": topic, "limit": 8}, default={})
        out.update(retrieved_context=res.get("context", ""), sources=res.get("sources", []),
                   asked_before=[q["question"] for q in bank.get("questions", [])])
        return out

    @staticmethod
    def _query(strategy: str, topic: str, state: InterviewState) -> str:
        skills = state.get("jd_skills") if strategy == "jd" else state.get("skills")
        return f"{topic} {' '.join((skills or [])[:5])} interview concepts".strip()

    async def generate_question(self, state: InterviewState) -> dict[str, Any]:
        idx = state.get("question_index", 0)
        item = state["plan"][idx]
        qtype, topic = item["type"], item["topic"]
        diff = item.get("difficulty") or state.get("difficulty", "medium")
        emit({"type": "node_start", "node": "generate_question", "message": "Writing your next question…"})
        prev = state.get("evaluations", [])
        follow = ((prev[-1] or {}).get("follow_up_question") if prev else "") or ""
        text = ""
        if qtype == "followup" and follow.strip():
            text = follow.strip()
            emit({"type": "token", "node": "generate_question", "text": text})
        else:
            project_hint = ""
            if qtype == "project":
                projects = state.get("resume_profile", {}).get("projects", [])
                if projects:
                    p = projects[(sum(1 for q in state.get("previous_questions", []) if q.get("type") == "project")) % len(projects)]
                    project_hint = f"{p.get('name')}: {p.get('description')} (tech: {', '.join(p.get('technologies', []))})"
            weak_note = await self.d.memory.prompt_note(state["user_id"], topic)
            avoid = [q["text"] for q in state.get("previous_questions", [])] + state.get("asked_before", [])
            messages = prompts.question_messages(
                role=state.get("role", ""), topic=topic, difficulty=diff, qtype=qtype,
                context=state.get("retrieved_context", ""), project_hint=project_hint, avoid=avoid,
                weak_note=weak_note, extra_explain=bool(state.get("needs_extra_explanation")),
                focus_note=state.get("plan_feedback", ""),
                jd_note=", ".join(state.get("jd_skills", [])[:10]) if state["mode"] == "jd_based" else "")
            async for piece in self.d.llm.astream_text(messages):
                text += piece
                emit({"type": "token", "node": "generate_question", "text": piece})
        text = text.strip().strip('"').strip()
        if not text:
            text = f"Explain the key concepts of {topic} and when you would apply them."
        q = {"id": new_id(), "text": text, "topic": topic, "difficulty": diff, "type": qtype,
             "index": idx, "sources": state.get("sources", [])}
        return {"current_question": q, "current_topic": topic,
                "messages": [{"role": "assistant", "kind": "question", "content": text}]}

    async def interviewer(self, state: InterviewState) -> dict[str, Any]:  # HITL: wait for the user
        q = state["current_question"]
        decision = interrupt({"kind": "await_answer", "question": q, "index": state.get("question_index", 0),
                              "total": len(state.get("plan", [])), "difficulty": q.get("difficulty")})
        action = decision.get("action")
        if action == "end":
            return {"end_requested": True}
        reset = {"feedback_decision": "", "reevaluate": False, "eval_revision": 0, "evaluation": {}, "feedback_text": ""}
        if action == "skip":
            return {**reset, "skipped": True, "user_answer": "", "extra_context": "",
                    "messages": [{"role": "user", "kind": "skip", "content": "(skipped)"}]}
        answer = str(decision.get("answer", "")).strip()
        return {**reset, "skipped": False, "user_answer": answer,
                "extra_context": str(decision.get("additional_context") or ""),
                "messages": [{"role": "user", "kind": "answer", "content": answer}]}

    async def record_skip(self, state: InterviewState) -> dict[str, Any]:
        q = {**state["current_question"], "skipped": True}
        await self.d.repo.save_answer(
            interview_id=state["interview_id"], user_id=state["user_id"], index=q["index"], question=q,
            answer="", skipped=True, evaluation=None, feedback_text="", decision="skip")
        return {"previous_questions": state.get("previous_questions", []) + [q],
                "answers": state.get("answers", []) + [{"index": q["index"], "skipped": True}],
                "evaluations": state.get("evaluations", []) + [None],
                "question_index": state.get("question_index", 0) + 1, "skipped": False}

    async def evaluate_answer(self, state: InterviewState) -> dict[str, Any]:
        q, uid = state["current_question"], state["user_id"]
        emit({"type": "node_start", "node": "evaluate_answer", "message": "Evaluating your answer…"})
        ctx = state.get("retrieved_context", "")
        if q["type"] not in ("behavioral",):
            query = f"{q['topic']} {q['text']} {state.get('user_answer', '')[:400]}"
            res = await self.d.mcp.call("search_technical_knowledge",
                                        {"user_id": uid, "query": query, "strategy": "evaluation"}, default={})
            ctx = res.get("context") or ctx
        ev = await self.d.evaluation.evaluate(
            question=q["text"], answer=state["user_answer"], topic=q["topic"], difficulty=q["difficulty"],
            context=ctx, extra_context=state.get("extra_context", ""))
        return {"evaluation": ev, "eval_revision": state.get("eval_revision", 0) + (1 if state.get("reevaluate") else 0),
                "reevaluate": False}

    async def feedback(self, state: InterviewState) -> dict[str, Any]:
        emit({"type": "node_start", "node": "feedback", "message": "Writing personalised feedback…"})
        q, ev = state["current_question"], state["evaluation"]
        note = await self.d.memory.prompt_note(state["user_id"], q["topic"])
        text = ""
        try:
            async for piece in self.d.llm.astream_text(prompts.feedback_messages(
                    question=q["text"], answer=state["user_answer"], evaluation=ev, memory_note=note,
                    difficulty=q["difficulty"])):
                text += piece
                emit({"type": "token", "node": "feedback", "text": piece})
        except Exception as exc:
            if text:
                log.warning("feedback stream interrupted: %s", exc)
            else:
                text = self._plain_feedback(ev)
                emit({"type": "token", "node": "feedback", "text": text})
        return {"feedback_text": text.strip(),
                "messages": [{"role": "assistant", "kind": "feedback", "content": text.strip()}]}

    @staticmethod
    def _plain_feedback(ev: dict[str, Any]) -> str:
        bits = [f"Score: {ev['score']}/10."]
        if ev.get("strengths"):
            bits.append("Good: " + "; ".join(ev["strengths"][:2]) + ".")
        if ev.get("missing_points"):
            bits.append("Missing: " + "; ".join(ev["missing_points"][:3]) + ".")
        return " ".join(bits)

    async def review_feedback(self, state: InterviewState) -> dict[str, Any]:  # HITL
        decision = interrupt({"kind": "review_feedback", "question": state["current_question"],
                              "evaluation": state["evaluation"], "feedback_text": state.get("feedback_text", ""),
                              "eval_revision": state.get("eval_revision", 0)})
        action = decision.get("action")
        if action == "end":
            return {"end_requested": True}
        if action == "reevaluate":
            note = str(decision.get("note") or "").strip()
            merged = (state.get("extra_context", "") + "\n" + note).strip()[:3000]
            return {"reevaluate": True, "extra_context": merged, "feedback_decision": "reevaluate"}
        return {"reevaluate": False, "feedback_decision": "reject" if action == "reject" else "accept"}

    async def update_performance(self, state: InterviewState) -> dict[str, Any]:
        q, ev = state["current_question"], dict(state["evaluation"])
        decision = state.get("feedback_decision") or "accept"
        if decision == "reevaluate":  # cap reached -> keep latest evaluation
            decision = "accept"
        rejected = decision == "reject"
        ev["rejected"] = rejected
        await self.d.repo.save_answer(
            interview_id=state["interview_id"], user_id=state["user_id"], index=q["index"], question=q,
            answer=state["user_answer"], skipped=False, evaluation=ev,
            feedback_text=state.get("feedback_text", ""), decision=decision)
        scores = list(state.get("scores", []))
        if not rejected:
            await self.d.memory.record_answer(state["user_id"], q["topic"], ev["score"], q["difficulty"],
                                              ev.get("missing_points"))
            scores.append(ev["score"])
        return {"previous_questions": state.get("previous_questions", []) + [q],
                "answers": state.get("answers", []) + [{"index": q["index"], "answer": state["user_answer"]}],
                "evaluations": state.get("evaluations", []) + [ev], "scores": scores,
                "question_index": state.get("question_index", 0) + 1}

    async def adjust_difficulty(self, state: InterviewState) -> dict[str, Any]:
        new, reason, extra = self.d.personalization.adjust_difficulty(
            state.get("difficulty", "medium"), state.get("scores", []))
        emit({"type": "status", "node": "adjust_difficulty", "message": f"Difficulty: {new} ({reason})"})
        return {"difficulty": new, "needs_extra_explanation": extra,
                "difficulty_history": state.get("difficulty_history", []) + [new]}

    # ---------------------------------------------------------------- finish
    async def summary(self, state: InterviewState) -> dict[str, Any]:
        emit({"type": "node_start", "node": "summary", "message": "Preparing your summary…"})
        stats = compute_stats(state)
        text = ""
        if stats["answered"] == 0:
            text = "The interview ended before any answers were scored. Start another one whenever you are ready."
            emit({"type": "token", "node": "summary", "text": text})
        else:
            try:
                async for piece in self.d.llm.astream_text(prompts.summary_messages(stats)):
                    text += piece
                    emit({"type": "token", "node": "summary", "text": piece})
            except Exception as exc:
                log.warning("summary narrative failed: %s", exc)
                if not text:
                    text = (f"You answered {stats['answered']} question(s) with an average of {stats['avg_score']}/10. "
                            f"Strongest: {', '.join(stats['strongest_topics']) or 'n/a'}. "
                            f"Focus next on: {', '.join(stats['weakest_topics']) or 'keep practising'}.")
                    emit({"type": "token", "node": "summary", "text": text})
        summary = {**stats, "narrative": text.strip()}
        await self.d.repo.save_summary(interview_id=state["interview_id"], user_id=state["user_id"],
                                       summary=summary, role=state.get("role"))
        await self.d.memory.record_interview(state["user_id"], state["interview_id"], stats["avg_score"],
                                             stats["topic_scores"], state.get("role"))
        return {"summary": summary, "status": "ended" if state.get("end_requested") else "completed",
                "messages": [{"role": "assistant", "kind": "summary", "content": text.strip()}]}
