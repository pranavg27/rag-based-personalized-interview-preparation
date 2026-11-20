from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.enums import JOB_ROLES, Difficulty, InterviewMode

Decision = Literal["approve", "reject", "accept", "reevaluate", "skip", "retry"]


class HITLConfig(BaseModel):
    skills: bool = True
    plan: bool = True
    feedback: bool = True


class InterviewStartRequest(BaseModel):
    mode: InterviewMode
    role: str | None = None
    skills: list[str] = Field(default_factory=list, max_length=20)
    resume_id: str | None = None
    jd_document_id: str | None = None
    job_description: str | None = Field(default=None, max_length=30000)
    difficulty: Difficulty | None = None
    max_questions: int = Field(default=5, ge=1, le=20)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    stream: bool = False

    @field_validator("role")
    @classmethod
    def _role(cls, v: str | None) -> str | None:
        if v is not None and v not in JOB_ROLES:
            raise ValueError(f"role must be one of {JOB_ROLES}")
        return v

    @field_validator("skills")
    @classmethod
    def _skills(cls, v: list[str]) -> list[str]:
        return [s.strip()[:60] for s in v if s and s.strip()]

    @model_validator(mode="after")
    def _mode_requirements(self) -> "InterviewStartRequest":
        if self.mode == InterviewMode.skill_based and not self.skills:
            raise ValueError("skill_based interviews need at least one skill")
        if self.mode == InterviewMode.resume_based and not self.resume_id:
            raise ValueError("resume_based interviews need resume_id")
        if self.mode == InterviewMode.jd_based and not (self.jd_document_id or self.job_description):
            raise ValueError("jd_based interviews need jd_document_id or job_description")
        return self


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=8000)
    additional_context: str | None = Field(default=None, max_length=2000)
    stream: bool = False


class PlanItemIn(BaseModel):
    topic: str = Field(min_length=1, max_length=80)
    type: Literal["technical", "project", "followup", "behavioral"] = "technical"
    difficulty: Difficulty | None = None


class NextRequest(BaseModel):
    decision: Decision
    skills: list[str] | None = Field(default=None, max_length=30)
    plan: list[PlanItemIn] | None = Field(default=None, max_length=20)
    note: str | None = Field(default=None, max_length=2000)
    stream: bool = False


class ControlRequest(BaseModel):
    stream: bool = False


class QueuedResponse(BaseModel):
    queued: bool
    stream_url: str | None = None
    interview: dict[str, Any]
