"""Typed LangGraph state. Everything is JSON-serialisable so it can live in the Mongo checkpoint."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class InterviewState(TypedDict, total=False):
    # identity
    interview_id: str
    user_id: str
    thread_id: str
    # configuration
    mode: str
    role: str
    resume_id: str
    requested_skills: list[str]
    jd_text: str
    requested_difficulty: str
    max_questions: int
    hitl: dict[str, bool]
    # analysis / planning
    resume_profile: dict[str, Any]
    skills: list[str]
    jd_skills: list[str]
    jd_analysis: dict[str, Any]
    weak_topics: list[str]
    plan: list[dict[str, Any]]
    plan_rejected: bool
    plan_feedback: str
    plan_revision: int
    # short-term interview memory
    question_index: int
    current_topic: str
    difficulty: str
    difficulty_history: list[str]
    needs_extra_explanation: bool
    current_question: dict[str, Any]
    previous_questions: list[dict[str, Any]]
    answers: list[dict[str, Any]]
    evaluations: list[dict[str, Any]]
    scores: list[float]
    messages: Annotated[list[dict[str, Any]], operator.add]
    # retrieval
    retrieved_context: str
    sources: list[dict[str, Any]]
    asked_before: list[str]
    # per-turn
    user_answer: str
    extra_context: str
    skipped: bool
    evaluation: dict[str, Any]
    eval_revision: int
    feedback_text: str
    feedback_decision: str
    reevaluate: bool
    # control
    end_requested: bool
    summary: dict[str, Any]
    status: str
