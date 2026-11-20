"""Resume parsing (PDF text + LLM structured extraction with a heuristic fallback)."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader

from app.database import mongo as m
from app.errors import NotFoundError, ValidationFailed
from app.services import prompts
from app.services.llm_service import LLMService
from app.utils.ids import new_id, utcnow

log = logging.getLogger(__name__)

SKILL_VOCAB = [
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "Go", "Rust", "Kotlin", "SQL", "NoSQL", "MongoDB",
    "PostgreSQL", "MySQL", "Redis", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git", "Linux", "FastAPI", "Flask",
    "Django", "Spring Boot", "REST", "GraphQL", "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "PyTorch", "TensorFlow", "scikit-learn", "Pandas", "NumPy", "LangChain", "LangGraph", "RAG", "LLM",
    "Transformers", "Hugging Face", "ChromaDB", "FAISS", "Streamlit", "Data Structures", "Algorithms", "DSA",
    "DBMS", "Operating Systems", "Computer Networks", "System Design", "Android", "Jetpack Compose", "Kotlin Coroutines",
]


def _list_of_dicts(v: Any, key: str) -> list[dict]:
    out = []
    for item in v or []:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str) and item.strip():
            out.append({key: item.strip()})
    return out


class ResumeProfile(BaseModel):
    name: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    experience: list[dict[str, Any]] = Field(default_factory=list)
    education: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("name", "summary", mode="before")
    @classmethod
    def _s(cls, v):
        return v if isinstance(v, str) else ""

    @field_validator("skills", mode="before")
    @classmethod
    def _skills(cls, v):
        if isinstance(v, str):
            v = re.split(r"[,;\n]", v)
        return [str(s).strip() for s in (v or []) if str(s).strip()]

    @field_validator("projects", mode="before")
    @classmethod
    def _p(cls, v):
        return _list_of_dicts(v, "name")

    @field_validator("experience", mode="before")
    @classmethod
    def _e(cls, v):
        return _list_of_dicts(v, "role")

    @field_validator("education", mode="before")
    @classmethod
    def _ed(cls, v):
        return _list_of_dicts(v, "degree")


def extract_text_from_pdf(path: Path, max_chars: int = 60000) -> str:
    try:
        reader = PdfReader(str(path))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as exc:
        raise ValidationFailed(f"Could not read PDF: {exc}") from exc
    text = text.strip()
    if len(text) < 30:
        raise ValidationFailed("The PDF has no extractable text (scanned image?).")
    return text[:max_chars]


def heuristic_skills(text: str) -> list[str]:
    found = []
    for skill in SKILL_VOCAB:
        if re.search(r"(?<![A-Za-z0-9+#])" + re.escape(skill) + r"(?![A-Za-z0-9+#])", text, re.I):
            found.append(skill)
    return found


def heuristic_profile(text: str) -> ResumeProfile:
    sections: dict[str, list[str]] = {}
    current = None
    heads = {"project": "projects", "experience": "experience", "work": "experience", "education": "education"}
    for line in text.splitlines():
        low = line.strip().lower().rstrip(":")
        if low and len(low) < 30 and any(low.startswith(h) for h in heads):
            current = next(v for k, v in heads.items() if low.startswith(k))
            sections.setdefault(current, [])
        elif low and len(low) < 30 and low in {"skills", "technical skills", "summary", "certifications", "achievements"}:
            current = None
        elif current:
            sections[current].append(line.strip())

    def blocks(lines: list[str], key: str) -> list[dict]:
        items, buf = [], []
        for ln in lines + [""]:
            if ln:
                buf.append(ln)
            elif buf:
                items.append({key: buf[0][:120], "description": " ".join(buf[1:])[:500]})
                buf = []
        return items[:8]

    return ResumeProfile(
        summary=text[:300].replace("\n", " "), skills=heuristic_skills(text),
        projects=blocks(sections.get("projects", []), "name"),
        experience=blocks(sections.get("experience", []), "role"),
        education=blocks(sections.get("education", []), "degree"),
    )


def dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        k = i.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(i.strip())
    return out


class ResumeService:
    def __init__(self, llm: LLMService):
        self.llm = llm

    @property
    def col(self):
        return m.get_db()[m.RESUMES]

    async def extract_profile(self, text: str) -> tuple[ResumeProfile, str]:
        base = heuristic_profile(text)
        try:
            system, user = prompts.resume_messages(text)
            prof = await self.llm.ajson(system, user, ResumeProfile)
            prof.skills = dedupe(prof.skills + base.skills)
            if not prof.projects:
                prof.projects = base.projects
            return prof, "llm"
        except Exception as exc:
            log.warning("LLM resume extraction failed (%s); using heuristic parser", exc)
            return base, "heuristic"

    async def create(self, *, user_id: str, filename: str, stored_path: str, size: int, text: str,
                     profile: ResumeProfile, extraction: str, target_role: str | None) -> dict:
        doc = {"_id": new_id(), "user_id": user_id, "filename": filename, "stored_path": stored_path,
               "size_bytes": size, "text": text, "profile": profile.model_dump(), "extraction": extraction,
               "target_role": target_role, "uploaded_at": utcnow()}
        await self.col.insert_one(doc)
        return doc

    async def list(self, user_id: str) -> list[dict]:
        return await self.col.find({"user_id": user_id}).sort("uploaded_at", -1).to_list(100)

    async def get(self, user_id: str, resume_id: str) -> dict:
        doc = await self.col.find_one({"_id": resume_id, "user_id": user_id})
        if not doc:
            raise NotFoundError("Resume not found.")
        return doc

    async def delete(self, user_id: str, resume_id: str) -> dict:
        doc = await self.get(user_id, resume_id)
        await self.col.delete_one({"_id": resume_id, "user_id": user_id})
        return doc
