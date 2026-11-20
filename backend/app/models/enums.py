from __future__ import annotations

from enum import StrEnum


class InterviewMode(StrEnum):
    resume_based = "resume_based"
    skill_based = "skill_based"
    jd_based = "jd_based"
    mock_interview = "mock_interview"
    weak_topics = "weak_topics"


class Difficulty(StrEnum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


DIFFICULTY_LADDER = ["easy", "medium", "hard"]

JOB_ROLES = [
    "AI/ML Engineer",
    "Machine Learning Engineer",
    "Python Developer",
    "Java Developer",
    "Backend Developer",
    "Software Development Engineer",
    "Data Scientist",
    "Generative AI Engineer",
    "Agentic AI Engineer",
]

SKILL_OPTIONS = [
    "DSA", "DBMS", "Operating Systems", "Computer Networks", "Java", "Python",
    "Machine Learning", "Deep Learning", "NLP", "RAG", "LangChain", "LangGraph",
    "FastAPI", "System Design",
]

ROLE_TOPICS: dict[str, list[str]] = {
    "AI/ML Engineer": ["Machine Learning", "Deep Learning", "Python", "NLP", "System Design"],
    "Machine Learning Engineer": ["Machine Learning", "Deep Learning", "Python", "DSA", "System Design"],
    "Python Developer": ["Python", "DSA", "DBMS", "FastAPI", "Operating Systems"],
    "Java Developer": ["Java", "DSA", "DBMS", "Operating Systems", "System Design"],
    "Backend Developer": ["DBMS", "FastAPI", "System Design", "Computer Networks", "DSA"],
    "Software Development Engineer": ["DSA", "System Design", "DBMS", "Operating Systems", "Computer Networks"],
    "Data Scientist": ["Machine Learning", "Python", "DBMS", "Deep Learning", "NLP"],
    "Generative AI Engineer": ["RAG", "LangChain", "NLP", "Deep Learning", "Python"],
    "Agentic AI Engineer": ["LangGraph", "LangChain", "RAG", "Python", "System Design"],
}

PLAN_TYPES = {"technical", "project", "followup", "behavioral"}
