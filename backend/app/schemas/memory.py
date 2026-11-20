from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.models.enums import JOB_ROLES, Difficulty


class PreferencesIn(BaseModel):
    preferred_role: str | None = None
    preferred_difficulty: Difficulty | None = None
    default_questions: int | None = Field(default=None, ge=1, le=20)
    learning_preferences: str | None = Field(default=None, max_length=500)

    @field_validator("preferred_role")
    @classmethod
    def _role(cls, v: str | None) -> str | None:
        if v is not None and v not in JOB_ROLES:
            raise ValueError(f"role must be one of {JOB_ROLES}")
        return v
