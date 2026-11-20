"""Structured answer evaluation with a configurable rubric and a heuristic safety net."""
from __future__ import annotations

import logging
import re
from typing import Any

from langsmith import traceable
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.services import prompts
from app.services.llm_service import LLMService

log = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    score: float = Field(default=0, ge=0, le=10)
    max_score: float = 10
    correctness: str = ""
    missing_points: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    ideal_answer: str = ""
    follow_up_question: str = ""
    rubric: dict[str, float] = Field(default_factory=dict)
    degraded: bool = False

    @field_validator("missing_points", "strengths", "improvements", mode="before")
    @classmethod
    def _lst(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x) for x in (v or [])]

    @field_validator("score", mode="before")
    @classmethod
    def _score(cls, v):
        try:
            return max(0.0, min(10.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("rubric", mode="before")
    @classmethod
    def _rub(cls, v):
        out = {}
        for k, val in (v or {}).items() if isinstance(v, dict) else []:
            try:
                out[str(k)] = max(0.0, min(10.0, float(val)))
            except (TypeError, ValueError):
                continue
        return out


def weighted_score(rubric_scores: dict[str, float], weights: dict[str, float]) -> float | None:
    keys = [k for k in weights if k in rubric_scores]
    if not keys:
        return None
    total_w = sum(weights[k] for k in keys)
    return sum(weights[k] * rubric_scores[k] for k in keys) / total_w


def heuristic_evaluation(answer: str, context: str, topic: str) -> EvaluationResult:
    """Used only if the LLM output is unusable: coarse keyword-overlap score, clearly flagged."""
    words = {w for w in re.findall(r"[a-z0-9+#]{4,}", (context or topic).lower())}
    ans = set(re.findall(r"[a-z0-9+#]{4,}", answer.lower()))
    overlap = len(words & ans) / max(len(words), 1) if words else min(len(ans) / 60, 1)
    score = round(min(10.0, 2 + overlap * 8 + min(len(answer) / 400, 1)), 1)
    return EvaluationResult(
        score=score, correctness="Automatic fallback scoring was used because the evaluator was unavailable.",
        improvements=["Add more precise technical detail and examples."], degraded=True,
        ideal_answer="(unavailable)", follow_up_question="")


class EvaluationService:
    def __init__(self, llm: LLMService, settings: Settings | None = None):
        self.llm, self.s = llm, settings or get_settings()

    @traceable(name="evaluate_answer", run_type="chain")
    async def evaluate(self, *, question: str, answer: str, topic: str, difficulty: str, context: str = "",
                       extra_context: str = "") -> dict[str, Any]:
        rubric = self.s.eval_rubric
        system, user = prompts.evaluation_messages(
            question=question, answer=answer, topic=topic, difficulty=difficulty, context=context,
            extra_context=extra_context, rubric=rubric)
        try:
            res = await self.llm.ajson(system, user, EvaluationResult)
            blended = weighted_score(res.rubric, rubric)
            if blended is not None:
                res.score = round(blended, 1)
        except Exception as exc:
            log.warning("evaluation fell back to heuristic: %s", exc)
            res = heuristic_evaluation(answer, context, topic)
        res.score = round(max(0.0, min(10.0, res.score)), 1)
        res.max_score = 10
        return res.model_dump()
