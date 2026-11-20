"""Conditional routing functions. Pure functions of state -> node name (unit-testable)."""
from __future__ import annotations

from app.graph.state import InterviewState

MAX_REEVALUATIONS = 3


def _end(state: InterviewState) -> bool:
    return bool(state.get("end_requested"))


def after_extract(state: InterviewState) -> str:
    hitl = state.get("hitl", {})
    if hitl.get("skills", True) and state.get("mode") in ("resume_based", "jd_based", "mock_interview"):
        return "review_skills"
    return "plan_interview"


def after_skill_review(state: InterviewState) -> str:
    return "summary" if _end(state) else "plan_interview"


def after_plan(state: InterviewState) -> str:
    return "approve_plan" if state.get("hitl", {}).get("plan", True) else "retrieve_knowledge"


def after_plan_approval(state: InterviewState) -> str:
    if _end(state):
        return "summary"
    return "plan_interview" if state.get("plan_rejected") else "retrieve_knowledge"


def after_interviewer(state: InterviewState) -> str:
    if _end(state):
        return "summary"
    return "record_skip" if state.get("skipped") else "evaluate_answer"


def after_feedback(state: InterviewState) -> str:
    return "review_feedback" if state.get("hitl", {}).get("feedback", True) else "update_performance"


def after_feedback_review(state: InterviewState) -> str:
    if _end(state):
        return "summary"
    if state.get("reevaluate") and state.get("eval_revision", 0) < MAX_REEVALUATIONS:
        return "evaluate_answer"
    return "update_performance"


def after_progress(state: InterviewState) -> str:
    if _end(state) or state.get("question_index", 0) >= len(state.get("plan", [])):
        return "summary"
    return "retrieve_knowledge"
