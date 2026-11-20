"""Implementations of every MCP tool. MCP servers wrap these; the MCP client also uses them as a
local fallback when a server is unreachable, so there is exactly one implementation per tool."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.database import mongo as m
from app.services import prompts
from app.services.container import get_services
from app.services.resume_service import SKILL_VOCAB, dedupe, heuristic_skills
from app.utils.ids import serialize


# ---------------- resume server ----------------------------------------------------------
async def extract_resume_skills(user_id: str, resume_id: str) -> dict[str, Any]:
    r = await get_services().resumes.get(user_id, resume_id)
    skills = r["profile"].get("skills", [])
    return {"resume_id": resume_id, "skills": skills, "count": len(skills)}


async def analyze_resume_projects(user_id: str, resume_id: str) -> dict[str, Any]:
    r = await get_services().resumes.get(user_id, resume_id)
    projects = []
    for p in r["profile"].get("projects", [])[:8]:
        techs = p.get("technologies") or heuristic_skills(f"{p.get('name','')} {p.get('description','')}")
        projects.append({"name": p.get("name", "Project"), "description": str(p.get("description", ""))[:500],
                         "technologies": techs, "probe_topics": techs[:3] or ["design decisions", "challenges"]})
    return {"resume_id": resume_id, "projects": projects}


async def summarize_resume(user_id: str, resume_id: str) -> dict[str, Any]:
    r = await get_services().resumes.get(user_id, resume_id)
    p = r["profile"]
    return {"resume_id": resume_id, "name": p.get("name", ""), "summary": p.get("summary", ""),
            "experience": p.get("experience", [])[:6], "education": p.get("education", [])[:4],
            "skill_count": len(p.get("skills", [])), "project_count": len(p.get("projects", []))}


# ---------------- question server -----------------------------------------------------------
class _Q(BaseModel):
    question: str
    topic: str = ""
    difficulty: str = "medium"
    expected_points: list[str] = Field(default_factory=list)


class _QList(BaseModel):
    questions: list[_Q]


async def generate_interview_questions(user_id: str, topic: str, difficulty: str = "medium", count: int = 3,
                                       role: str | None = None) -> dict[str, Any]:
    system, user = prompts.batch_question_messages(topic, difficulty, count, role)
    res = await get_services().llm.ajson(system, user, _QList)
    qs = [{**q.model_dump(), "topic": q.topic or topic, "difficulty": q.difficulty or difficulty}
          for q in res.questions[:count]]
    return {"topic": topic, "questions": qs}


async def retrieve_interview_questions(user_id: str, topic: str, limit: int = 10) -> dict[str, Any]:
    from app.services.interview_repo import InterviewRepo

    return {"topic": topic, "questions": await InterviewRepo().asked_questions(user_id, topic, limit)}


async def search_technical_knowledge(user_id: str, query: str, strategy: str = "technical",
                                     skill_category: str | None = None, top_k: int | None = None) -> dict[str, Any]:
    res = await get_services().retriever.retrieve(user_id, query, strategy, skill_category, top_k)
    return res.as_dict()


# ---------------- analytics server ------------------------------------------------------------
async def get_user_performance(user_id: str) -> dict[str, Any]:
    return await get_services().analytics.performance(user_id)


async def get_weak_topics(user_id: str, limit: int = 5) -> dict[str, Any]:
    return {"weak_topics": await get_services().analytics.weak_topics(user_id, limit)}


async def get_interview_history(user_id: str, limit: int = 10) -> dict[str, Any]:
    return {"interviews": await get_services().analytics.history(user_id, limit)}


async def generate_progress_report(user_id: str) -> dict[str, Any]:
    svc = get_services()
    stats = await svc.analytics.progress_stats(user_id)
    try:
        narrative = await svc.llm.ainvoke_text(prompts.progress_report_messages(stats))
    except Exception:
        narrative = stats["memory_summary"]
    return {"report": narrative, "stats": serialize(stats)}


# ---------------- job description server ----------------------------------------------------------
class JDAnalysis(BaseModel):
    role: str = ""
    seniority: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


def _jd_fallback(text: str) -> JDAnalysis:
    return JDAnalysis(required_skills=heuristic_skills(text),
                      responsibilities=[l.strip("-• ").strip() for l in text.splitlines() if l.strip()[:1] in "-•*"][:8])


async def analyze_job_description(user_id: str, job_description: str, role: str | None = None) -> dict[str, Any]:
    try:
        system, user = prompts.jd_messages(job_description, role)
        a = await get_services().llm.ajson(system, user, JDAnalysis)
        a.required_skills = dedupe(a.required_skills + heuristic_skills(job_description))
    except Exception:
        a = _jd_fallback(job_description)
    return a.model_dump()


async def extract_required_skills(job_description: str) -> dict[str, Any]:
    skills = heuristic_skills(job_description)
    extra = re.findall(r"(?:experience (?:with|in)|proficien\w+ (?:in|with)|knowledge of)\s+([A-Z][\w+#. /-]{2,30})", job_description)
    return {"skills": dedupe(skills + [e.strip(" .,") for e in extra])[:25]}


async def compare_resume_with_job_description(user_id: str, resume_id: str, job_description: str) -> dict[str, Any]:
    r = await get_services().resumes.get(user_id, resume_id)
    have = {s.lower(): s for s in r["profile"].get("skills", [])}
    need = (await extract_required_skills(job_description))["skills"]
    matched = [s for s in need if s.lower() in have]
    missing = [s for s in need if s.lower() not in have]
    return {"required": need, "matched": matched, "missing": missing,
            "match_score": round(100 * len(matched) / len(need), 1) if need else None}


LOCAL_TOOLS = {f.__name__: f for f in (
    extract_resume_skills, analyze_resume_projects, summarize_resume, generate_interview_questions,
    retrieve_interview_questions, search_technical_knowledge, get_user_performance, get_weak_topics,
    get_interview_history, generate_progress_report, analyze_job_description, extract_required_skills,
    compare_resume_with_job_description)}

TOOL_SERVER = {
    **dict.fromkeys(["extract_resume_skills", "analyze_resume_projects", "summarize_resume"], "resume"),
    **dict.fromkeys(["generate_interview_questions", "retrieve_interview_questions", "search_technical_knowledge"], "questions"),
    **dict.fromkeys(["get_user_performance", "get_weak_topics", "get_interview_history", "generate_progress_report"], "analytics"),
    **dict.fromkeys(["analyze_job_description", "extract_required_skills", "compare_resume_with_job_description"], "jd"),
}
