from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResumeOut(BaseModel):
    id: str
    filename: str
    size_bytes: int
    target_role: str | None = None
    extraction: str
    profile: dict[str, Any]
    uploaded_at: str


class DocumentOut(BaseModel):
    id: str
    name: str
    doc_type: str
    skill_category: str
    chunk_count: int
    size_bytes: int
    uploaded_at: str


class JobDescriptionIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=30, max_length=30000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    skill_category: str | None = Field(default=None, max_length=60)
    stream: bool = True
